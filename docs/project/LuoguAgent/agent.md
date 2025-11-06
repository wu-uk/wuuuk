# Agent

我们将爬虫整合一下，变成一个 `LuoguCrawlerAgent` 类，主要职责是爬取题目和题解。并顺便添加缓存功能，避免在测试阶段多次在爬取和解析上浪费时间。

创建一个 `AnalysisAgent` 类，用于接收题目数据，生成 prompt 喂给大模型，然后给出结构化输出。这是本项目的核心。

## LuoguCrawlerAgent：爬虫策略的执行官

在上一篇文章中，我们探讨了爬取洛谷的“混合动力”策略（`requests` + `bs4` + `Crawl4ai`）以及踩过的坑。今天，我们把这个策略封装成一个可重用、有状态的 `LuoguCrawlerAgent` 类。

这个 Agent 是整个爬虫的“大脑”和“指挥官”。它的核心职责是将之前零散的函数（如 `search_for_solution_urls` 和 `extract_solution_content`）编排成一个健壮的、自动化的流水线。

### 核心设计：指挥官与流水线

经过重构，`LuoguCrawlerAgent` 的职责非常清晰。它的公共入口只有一个：

```python
async def run(self, problem_id: str, max_solutions: int = 3) -> Dict[str, Any]:
```

这个 `run` 方法本身不执行任何爬虫，它只做三件事：**检查缓存、执行任务、保存缓存**。

#### 1\. 缓存优先 (Caching)

这是 Agent 最重要的功能之一。`run` 方法会首先调用 `_load_from_cache`。

如果它在本地（例如 `./luogu_cache/P4137.json`）找到了缓存文件，并且缓存的题解数量大于等于本次请求的 `max_solutions`，它会立刻返回数据，**不执行任何爬虫**。这极大地节省了时间和 LLM Token。当我们后续开发“分析 Agent”时，我们可以反复运行程序，而爬虫 Agent 会立即从本地返回数据，让我们能专注于调试 AI 的部分。

#### 2\. 爬取流水线 (Pipeline)

如果缓存未命中，`run` 方法会调用 `_execute_crawl_pipeline` 来执行实际的爬取工作。这个流水线就是基于我们昨天的策略。

1.  **并发启动:** 使用 `asyncio.gather` **同时**启动两个任务：
      * **任务A (`_fetch_problem_details`)**: 用 `Crawl4ai` 提取题目详情。
      * **任务B (`_search_solution_urls`)**: 用 `requests+bs4`（在 `run_in_executor` 中运行）搜索题解链接列表。
2.  **并发提取:** 拿到题解 URL 列表后，再次使用 `asyncio.gather` 并发爬取（`_fetch_solution_content`）`max_solutions` 篇题解的详细内容。
3.  **汇总:** 收集所有结果，打包成 `final_output` 字典。

#### 3\. 保存结果 (Saving)

爬取流水线返回结果后，`run` 方法会调用 `_save_to_cache`，使用 `aiofiles` 异步地将 `final_output` 写入 JSON 文件，供下一次运行使用。

### 总结

通过这种方式，我们将复杂的爬取逻辑封装成了一个“黑盒”。主程序（`main`）不再关心爬虫的内部细节，它只需要知道：

```python
# 只需要实例化，然后运行
agent = LuoguCrawlerAgent(crawler)
final_data = await agent.run(TEST_PROBLEM_ID, max_solutions=3)
```

`LuoguCrawlerAgent` 会自动处理缓存、并发和错误，交付一份干净的数据。

### 核心代码

```python
class LuoguCrawlerAgent:
    """
    负责洛谷题目和题解的爬取、提取和缓存。
    """

    def __init__(self, crawler: AsyncWebCrawler, cache_dir: str = CACHE_DIR):
        self.crawler = crawler
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"[CrawlerAgent] 初始化完成，缓存目录: {self.cache_dir}")

    # --- 1. 公共主入口 (指挥官) ---

    async def run(self, problem_id: str, max_solutions: int = 3) -> Dict[str, Any]:
        """
        [主入口] 执行完整的爬取和提取流程。
        """
        print(f"\n--- [CrawlerAgent] 启动 {problem_id} 任务 (max={max_solutions}) ---")

        # 1. 尝试从缓存加载
        cached_data = await self._load_from_cache(problem_id, max_solutions)
        if cached_data:
            print(f"--- [CrawlerAgent] {problem_id} 任务完成 (来自缓存) ---")
            return cached_data

        # 2. 缓存未命中，执行爬取流水线
        print(f"[CrawlerAgent] CACHE MISS: {problem_id}。开始实时爬取...")
        final_output = await self._execute_crawl_pipeline(problem_id, max_solutions)

        # 3. 仅在爬取成功时保存到缓存
        if "error" not in final_output.get("problem", {}):
            await self._save_to_cache(problem_id, final_output)
        else:
            print(f"[CrawlerAgent] 题目爬取失败，不保存缓存。")

        print(f"--- [CrawlerAgent] {problem_id} 任务完成 (实时爬取)。---")
        return final_output

    # --- 2. 缓存处理 (I/O) ---

    async def _load_from_cache(self, problem_id: str, max_solutions: int) -> Optional[Dict[str, Any]]:
        """(职责1) 尝试从本地 JSON 加载并验证缓存"""
        cache_file = os.path.join(self.cache_dir, f"{problem_id}.json")
        try:
            async with aiofiles.open(cache_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                cached_data = json.loads(content)

            # 验证缓存数据
            if 'problem' not in cached_data or "error" in cached_data['problem']:
                print(f"[CrawlerAgent] CACHE INVALID: {problem_id} 缓存损坏。")
                return None
            
            # 验证缓存数量
            cached_solutions_count = len(cached_data.get("solutions", []))
            if cached_solutions_count >= max_solutions:
                print(f"[CrawlerAgent] CACHE HIT: {problem_id} (含 {cached_solutions_count} 篇题解) 满足需求。")
                return cached_data
            else:
                print(f"[CrawlerAgent] CACHE PARTIAL: 缓存 {cached_solutions_count} 篇, 需求 {max_solutions}。")
                return None

        except FileNotFoundError:
            return None  # 缓存未命中，正常
        except Exception as e:
            print(f"[CrawlerAgent] CACHE ERROR: 加载 {cache_file} 失败 ({e})。")
            return None # 缓存读取失败，当作未命中处理

    async def _save_to_cache(self, problem_id: str, data: Dict[str, Any]):
        """(职责2) 异步保存结果到 JSON 缓存"""
        cache_file = os.path.join(self.cache_dir, f"{problem_id}.json")
        try:
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                content_to_save = json.dumps(data, indent=2, ensure_ascii=False)
                await f.write(content_to_save)
            print(f"[CrawlerAgent] CACHE SAVE: 成功保存 {problem_id} 到 {cache_file}")
        except Exception as e:
            print(f"[CrawlerAgent] CACHE SAVE FAILED: 写入 {cache_file} 失败: {e}")

    # --- 3. 核心爬取流水线 (干活的) ---

    async def _execute_crawl_pipeline(self, problem_id: str, max_solutions: int) -> Dict[str, Any]:
        """(职责3) 执行所有爬取、搜索和提取任务"""
        
        # 3.1. 并行执行“爬题目”和“搜题解列表”
        problem_task = self._fetch_problem_details(problem_id)
        search_task = self._search_solution_urls(problem_id)
        
        problem_result, solution_urls = await asyncio.gather(problem_task, search_task)
        
        final_output = {
            "problem": problem_result,
            "solutions": []
        }
        
        # 如果题目爬取失败，则中止
        if "error" in problem_result:
            print(f"[CrawlerAgent] 严重错误：题目 {problem_id} 爬取失败。任务中止。")
            return final_output

        # 如果没有题解，也提前返回
        if not solution_urls:
            print(f"[CrawlerAgent] 警告：没有找到 {problem_id} 的题解链接。")
            return final_output

        # 3.2. 并行提取 N 篇题解
        urls_to_fetch = solution_urls[:max_solutions]
        print(f"[CrawlerAgent] 准备从 {len(urls_to_fetch)} 个链接中提取题解内容...")
        
        solution_tasks = [self._fetch_solution_content(article) for article in urls_to_fetch]
        solution_results = await asyncio.gather(*solution_tasks)
        
        # 3.3. 清洗题解结果
        successful_solutions = []
        for i, res in enumerate(solution_results):
            if "error" not in res:
                successful_solutions.append(res)
            else:
                print(f"[CrawlerAgent] 提取 {urls_to_fetch[i]['url']} 失败: {res['error']}")
        
        final_output["solutions"] = successful_solutions
        print(f"[CrawlerAgent] 提取完成，成功 {len(successful_solutions)} / {len(urls_to_fetch)} 篇。")
        return final_output

    # --- 4. 子任务  ---

    async def _fetch_problem_details(self, problem_id: str) -> Dict[str, Any]:
        """[子任务] 爬取并提取单个题目的详细信息。"""
        url = f"https://www.luogu.com.cn/problem/{problem_id}"
        print(f"[CrawlerAgent.Problem] 正在爬取题目: {url}")
        
        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED, # 题目页可以缓存
            extraction_strategy=PROBLEM_STRATEGY,
            delay_before_return_html=1
        )
        result = await self.crawler.arun(url, config=config)
        
        if not (result.success and result.extracted_content):
            return {"error": "crawl4ai 爬取题目失败", "details": result.error_message}
            
        try:
            data_list = json.loads(result.extracted_content)
            if not data_list:
                return {"error": "LLM 从题目页返回了空列表"}
            
            problem_data = ProblemDetails.model_validate(data_list[0])
            print(f"[CrawlerAgent.Problem] 题目 {problem_id} 提取成功。")
            return problem_data.model_dump()
        
        except json.JSONDecodeError:
            return {"error": "LLM 返回了无效的 JSON", "raw": result.extracted_content}
        except Exception as e:
            return {"error": f"Pydantic 验证失败: {e}", "raw": result.extracted_content}

    async def _search_solution_urls(self, problem_id: str) -> List[Dict[str, str]]:
        """[子任务] 异步执行同步的 URL 搜索。"""
        loop = asyncio.get_running_loop()
        # 在一个单独的线程中运行同步的 blocking IO
        results = await loop.run_in_executor(
            None, self._sync_search_solutions, problem_id
        )
        return results

    def _sync_search_solutions(self, problem_id: str) -> List[Dict[str, str]]:
        """[子任务] 同步使用 requests 搜索题解 URL。"""
        search_url = f"https://www.luogu.com.cn/article?keyword={problem_id}&page=1"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        print(f"[CrawlerAgent.Search] 正在用 requests 搜索: {search_url} ...")
        
        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            article_links = soup.find_all('a', href=re.compile(r'^/article/[a-zA-Z0-9]{8}$'))
            
            if not article_links:
                print("[CrawlerAgent.Search] 失败：没有找到任何 /article/... 链接。")
                return []

            results = []
            seen_urls = set()
            for link in article_links:
                href = link.get('href')
                if href in seen_urls:
                    continue
                
                title = link.get_text(strip=True)
                if href and (problem_id in title or problem_id.lower() in title):
                    results.append({"title": title, "url": href})
                    seen_urls.add(href)

            print(f"[CrawlerAgent.Search] 成功：找到 {len(results)} 篇相关题解链接。")
            return results
            
        except requests.RequestException as e:
            print(f"[CrawlerAgent.Search] 异常: {e}")
            return []

    async def _fetch_solution_content(self, article: Dict[str, str]) -> Dict[str, Any]:
        """[子任务] 爬取并提取单篇题解的详细内容。"""
        full_url = f"https://www.luogu.com.cn{article['url']}"
        print(f"[CrawlerAgent.Solution] 正在提取: {full_url} ...")
        
        config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS, # 题解页不应缓存
            extraction_strategy=SOLUTION_STRATEGY,
            delay_before_return_html=1
        )
        result = await self.crawler.arun(full_url, config=config)
        
        if not (result.success and result.extracted_content):
             return {"error": "crawl4ai 爬取题解失败"}
             
        try:
            data_list = json.loads(result.extracted_content)
            if not data_list:
                return {"error": "LLM 从题解页返回了空列表"}
            
            raw_data = data_list[0]
            clean_data = SolutionDetails.model_validate(raw_data)
            
            if not clean_data.solution_text:
                return {"error": "LLM 返回了空内容"}
            
            # 附加上 title 和 url
            final_data = clean_data.model_dump()
            final_data['title'] = article['title']
            final_data['url'] = article['url']
            return final_data

        except json.JSONDecodeError:
            return {"error": "LLM 返回了无效的 JSON", "raw": result.extracted_content}
        except Exception as e:
            return {"error": f"Pydantic 验证失败: {e}", "raw": result.extracted_content}
```

---

## AnalysisAgent

> 调用 chatglm 的过程中我遇到了较多的困难，比如一开始我希望直接上 langchain 架构，但是遇到了 `with_structured_output` 不支持的问题。后面就尝试从 zhipu 的官方 api 开始调试，对着官方文档，费了点功夫调通了。还有个问题是我一开始调 api 非常的慢，一度让我以为是网络问题或者单纯卡住了，后面发现是默认打开了思考模式 emmmmm....

`AnalysisAgent` 是我们流水线的“大脑”。`LuoguCrawlerAgent` 负责“搜集食材”（爬取题目和题解），而 `AnalysisAgent` 则负责将这些原始、杂乱的“食材”加工成一道“米其林大餐”（一份高质量、综合的分析报告）。

它的核心职责是**调用大模型（ZhipuAI）**，并**健壮地处理模型的输出**。

### 核心设计：提示、解析与双重保存

与爬虫 Agent 类似，`AnalysisAgent` 也被重构成了一个清晰的流水线，由 `run` 方法统一指挥。

#### 1. 构建提示 (`_build_prompts`)

这是最关键的一步。我们动态构建两条 Prompt：
* **System Prompt**: 告诉 LLM 它是一个“算法竞赛金牌教练”，并把 `ProblemAnalysis` Pydantic 模型的 Schema（`model_fields`）喂给它，命令它必须按这个格式返回。
* **User Prompt**: 把 `LuoguCrawlerAgent` 爬来的 `problem_data` 和 `solutions_data` 两个大 JSON 块塞进去，作为 LLM 分析的“原材料”。

#### 2. LLM 调用 (`_stream_llm_response`)

这里我们直接使用了 ZhipuAI 的官方 API。如引言所说，一个关键优化是设置 `thinking={"type": "disabled"}`，这能显著加快响应速度。另外开启流式输出，让我们能实时看到“打字”过程。

#### 3. “双保险”解析 (`_parse_to_dict`)

LLM 的输出并不总是严格的 JSON。有时它返回的是标准 JSON（双引号），有时它返回的是 Python 字典。

`_parse_to_dict` 方法实现了一个“双保险”：
    1.  首先，用 `json.loads()` 尝试按标准 JSON 解析。
    2.  如果失败了（`JSONDecodeError`），它不会立即放弃，而是会用 `ast.literal_eval()` 再次尝试。`ast.literal_eval` 可以安全地将 Python 字典格式的字符串解析成 Python 对象，完美解决了 LLM“不守规矩”的问题。

#### 4. 双重保存 (`_save_..._result`)

在成功解析出 Python 字典后，Agent 会同时保存两种格式的文件：
1.  **`_analysis.json`**: 保存为标准 JSON 文件，用于后续的机器读取或程序调用。
2.  **`_analysis.md`**: 提取字典中的 `detailed_solution` 和 `sample_code` 等字段，放到markdown中，这个主要是给我看的。

通过这个流程，`AnalysisAgent` 确保了无论爬虫数据多乱、LLM 回答多“随意”，我们最终都能得到一份结构化、易于阅读的高质量分析报告。

### 核心代码

```python
class AnalysisAgent:
    """
    负责调用 LLM 分析问题数据，并处理和保存结果。
    会同时保存 .json (机器可读) 和 .md (人类可读) 两种格式。
    """
    
    def __init__(self):
        self.llm = ZhipuAiClient(ZHIPU_API_KEY, ZHIPU_BASE_URL)
        self.schema = ProblemAnalysis  
        self.analysis_cache_dir = ANALYSIS_CACHE_DIR
        os.makedirs(self.analysis_cache_dir, exist_ok=True)
        print(f"[Agent] 初始化完成 (ZhipuAI)，结果保存至 {self.analysis_cache_dir}")

    def _build_prompts(self, problem_data: Dict[str, Any], solutions_data: List[Dict[str, Any]]) -> Tuple[str, str]:
        """构建系统和用户提示词"""
        system_prompt = get_system_prompt(self.schema.model_fields)
        user_prompt = get_user_prompt(problem_data, solutions_data)
        return system_prompt, user_prompt

    def _stream_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        """调用 LLM 并流式获取完整响应"""
        print("[Agent] 正在流式接收 LLM 响应...")
        response = self.llm.chat.completions.create(
            model="glm-4.6", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            thinking={"type": "disabled"},
            stream=True
        )

        full_content = ""
        for chunk in response:
            if not chunk.choices: continue
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                full_content += delta.content
                print(delta.content, end="", flush=True) 
            
            if chunk.choices[0].finish_reason:
                print(f"\n\n[Agent] 完成原因: {chunk.choices[0].finish_reason}")
                if hasattr(chunk, 'usage') and chunk.usage:
                    print(f"[Agent] 令牌使用: 输入 {chunk.usage.prompt_tokens}, 输出 {chunk.usage.completion_tokens}")
        
        print("\n[Agent] LLM 响应接收完毕。")
        return full_content

    def _extract_json_string(self, raw_content: str) -> str:
        """从 LLM 的原始输出中提取 { ... } 块"""
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            print("[Agent] 已从 LLM 响应中提取 { } 块。")
            return match.group(0)
        
        print("[Agent] 无法提取 { } 块，将尝试解析完整响应。")
        return raw_content

    def _parse_to_dict(self, s: str) -> Optional[Dict[str, Any]]:
        """使用双保险 (json / ast) 解析字符串为字典"""
        try:
            parsed = json.loads(s)
            print("[Agent] 成功解析为 JSON。")
            return parsed
        except json.JSONDecodeError:
            print("[Agent] JSON 解析失败，尝试 ast.literal_eval (Python 字典)...")
            try:
                parsed = ast.literal_eval(s)
                print("[Agent] 成功解析为 Python 字面量。")
                return parsed
            except Exception as e:
                print(f"[Agent] 所有解析均失败: {e}")
                return None

    def _save_json_result(self, problem_id: str, data: Dict[str, Any]):
        """将解析成功的字典保存为美化的 .json 文件"""
        save_path = os.path.join(self.analysis_cache_dir, f"{problem_id}_analysis.json")
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[Agent] 分析结果 (JSON) 已保存到: {save_path}")
        except Exception as e:
            print(f"[Agent] 保存 .json 文件时出错: {e}")

    def _save_markdown_result(self, problem_id: str, data: Dict[str, Any]):
        """[!] 将解析结果中的关键字段提取并保存为 .md 文件"""
        save_path = os.path.join(self.analysis_cache_dir, f"{problem_id}_analysis.md")
        
        # 1. 从字典中提取内容
        solution_text = data.get('detailed_solution', 'LLM 未提供详细题解。')
        sample_code = data.get('sample_code', 'LLM 未提供示例代码。')
        keywords = data.get('keywords', [])
        
        # 我们的 Pydantic 描述里写了是 C++
        code_lang = "cpp"

        # 2. 构建 Markdown 字符串
        md_content = f"# {problem_id} 详细题解\n\n"
        md_content += f"{solution_text}\n\n"
        
        md_content += f"# 参考代码 ({code_lang})\n\n"
        md_content += f"```{code_lang}\n"
        md_content += f"{sample_code}\n"
        md_content += "```\n\n"
        
        md_content += "# 核心知识点\n\n"
        if keywords:
            for keyword in keywords:
                md_content += f"* {keyword}\n"
        else:
            md_content += "无\n"

        # 3. 保存文件
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            print(f"[Agent] 分析结果 (Markdown) 已保存到: {save_path}")
        except Exception as e:
            print(f"[Agent] 保存 .md 文件时出错: {e}")

    def _save_error(self, problem_id: str, raw_content: str):
        """保存原始的、无法解析的 LLM 响应"""
        error_path = os.path.join(self.analysis_cache_dir, f"{problem_id}_error.txt")
        try:
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(raw_content)
            print(f"[Agent] 原始错误响应已保存到: {error_path}")
        except Exception as e:
            print(f"[Agent] 保存 _error.txt 文件时出错: {e}")

    def run(self, problem_id: str, problem_data: Dict[str, Any], solutions_data: List[Dict[str, Any]]) -> str:
        """
        [主入口] 执行完整的分析、解析和保存流程。
        返回 LLM 的原始字符串响应。
        """
        print(f"\n--- [Agent] 开始分析 {problem_id} ---")
        
        # 1. 构建提示
        system_prompt, user_prompt = self._build_prompts(problem_data, solutions_data)
        
        # 2. 获取 LLM 响应
        full_content = self._stream_llm_response(system_prompt, user_prompt)
        if not full_content:
            print("[Agent] LLM 响应为空，任务中止。")
            return full_content

        # 3. 提取
        extracted_str = self._extract_json_string(full_content)
        
        # 4. 解析
        parsed_data = self._parse_to_dict(extracted_str)

        # 5. 保存
        if parsed_data:
            # [!] 同时调用两个保存方法
            self._save_json_result(problem_id, parsed_data)
            self._save_markdown_result(problem_id, parsed_data)
        else:
            self._save_error(problem_id, full_content)
            
        return full_content
```

## 效果

值得一提的是，在调通之后，我尝试把模型的输出拿去提交，以luogu紫题P4137为例，在5次测试中，只有一次代码是可以交上去直接AC的。在P1117这道题甚至出现了编译错误（尽管后面我看了一下是个小错误）。这说明不开思考模式的能力还是比较有限的。但是从题解的质量上来说结构上是比较合理的，也比人类的题解相对来说要更加详细。

另外，我发现在爬虫阶段爬取到的前三条并非是最高赞的回答，这是一个爬虫策略的遗留问题：在solution/里面题解确实是按照点赞数排序的，但是article里并非如此。另外像P4137这种题目有多个解法的可能对LLM并不是很友好。如果能同一种解法多个题解可能效果更好。这方面还是可以继续优化的。