# SQL 语言

## 1\. DDL (数据定义语言)

### 1.1 创建表 (CREATE TABLE)

```sql
create table instructor (
    ID          char(5),
    name        varchar(20) not null,
    dept_name   varchar(20),
    salary      numeric(8,2),
    primary key (ID),
    foreign key (dept_name) references department,
    check(salary > 0)
);

create table section (
    course_id   varchar(8),
    sec_id      varchar(8),
    semester    varchar(6),
    year        numeric(4, 0),
    building    varchar(15),
    room_number varchar(7),
    time_slot_id varchar(4),
    primary key(course_id, sec_id, semester, year),
    foreign key(course_id) references course,
    check (semester in ('Fall', 'Winter', 'Spring', 'Summer'))
);
```

### 1.2 删除表 (DROP TABLE)

删除表的所有信息（元组和模式）。

```sql
drop table r;
```

### 1.3 修改表 (ALTER TABLE)

向已有关系中添加或删除属性。

```sql
/* 添加属性，所有元组在新属性上默认为 null */
alter table r add A D;

/* 删除属性 (某些数据库可能不支持) */
alter table r drop A;
```

### 1.4 创建索引 (CREATE INDEX)

创建索引以加快查询速度。

```sql
create index studentID_index on student(ID);
```

## 2\. DML (数据操纵语言)

### 2.1 插入 (INSERT)

#### 在表中插入元素

```sql
insert into course
	values('CS-347', 'Database Systems', 'Comp. Sci.', 4);
```

等价于：

```sql
insert into course(course_id, title, dept_name, credits)
	values('CS-347', 'Database Systems', 'Comp. Sci.', 4);
```

#### 在表中插入查询结果

```sql
/* 将所有讲师添加到学生关系中，学分设为 0 */
insert into student
	select ID, name, dept_name, 0
	from instructor;
```

### 2.2 删除 (DELETE)

从关系 r 中删除满足谓词 P 的元组。

```sql
delete from r
where P;
```

例如，删除工资低于平均工资的讲师：

```sql
delete from instructor
where salary < (select avg(salary)
               	from instructor);
```

### 2.3 修改 (UPDATE)

#### 全部修改

```sql
update instructor
set salary = salary * 1.05;
```

#### 部分修改

```sql
update instructor
set salary = salary * 1.03
where salary > 100000;
```

#### 条件修改 (CASE)

```sql
update instructor
set salary = 	case
					when salary <= 100000 then salary * 1.05
					else salary * 1.03
				end;
```

#### 基于标量子查询的修改

为所有学生重新计算并更新 `tot_cred` (总学分)。

```sql
update student S
set tot_cred = (
	select sum(credits)
    from takes, course
    where takes.course_id = course.course_id and
          S.ID = takes.ID and
          takes.grade <> 'F' and takes.grade is not null
);
```

## 3\. DQL (数据查询语言)

### 3.1 `SELECT` 基本结构

```sql
select  A1, A2, ..., An
from  r1, r2, ..., rm
where P;
```

#### `SELECT` 子句

  * `distinct`：去除结果中的重复行。
    ```sql
    select distinct dept_name
    from instructor;
    ```
  * `*`：表示“所有属性”。
    ```sql
    select *
    from instructor;
    ```
  * `as`：重命名属性或表达式的结果。
    ```sql
    select ID, name, salary/12 as monthly_salary
    from instructor;
    ```

#### `WHERE` 子句

  * `between`：用于范围查询。
    ```sql
    select name
    from instructor
    where salary between 90000 and 100000;
    ```
  * `like`：用于字符串模糊匹配。
      * `%`：匹配任意子串。
      * `_`：匹配任意一个字符。
    <!-- end list -->
    ```sql
    select name
    from instructor
    where name like '%dar%';
    ```
  * `is null`：检查空值。
    ```sql
    select name
    from instructor
    where salary is null;
    ```

#### `ORDER BY` 子句

用于对结果元组进行排序。

```sql
/* 默认升序 (asc) */
select distinct name
from instructor
order by name;

/* 降序 (desc) */
order by name desc;
```

### 3.2 聚合函数 (Aggregate Functions)

对一组值进行操作并返回单个值。

  * `avg`: 平均值
  * `min`: 最小值
  * `max`: 最大值
  * `sum`: 合计
  * `count`: 计数

<!-- end list -->

```sql
/* 找出 Comp. Sci. 系讲师的平均工资 */
select avg (salary)
from instructor
where dept_name = 'Comp. Sci.';

/* 统计课程总数 */
select count (*)
from course;

/* 统计 2010 年春季开课的教师人数 (去重) */
select count (distinct ID)
from teaches
where semester = 'Spring' and year = 2010;
```

#### `GROUP BY` 子句

与聚合函数一起使用，将元组按属性值分组。

```sql
select dept_name, avg (salary) as avg_salary
from instructor
group by dept_name;
```

> **注意**：`SELECT` 子句中**未被**聚合的属性，必须出现在 `GROUP BY` 子句中。

#### `HAVING` 子句

用于对 `GROUP BY` 形成的分组进行条件筛选。

```sql
select dept_name, avg (salary)
from instructor
group by dept_name
having avg (salary) > 42000;
```

> **区别**：`WHERE` 在分组前过滤行，`HAVING` 在分组后过滤组。

### 3.3 集合运算 (Set Operations)

  * `union`：并集（自动去重）
  * `intersect`：交集（自动去重）
  * `except`：差集（自动去重）
  * (使用 `all` 关键字可保留重复项，如 `union all`)

<!-- end list -->

```sql
/* 2009 年秋季或 2010 年春季开设的课程 */
(select course_id from section where sem = 'Fall' and year = 2009)
union
(select course_id from section where sem = 'Spring' and year = 2010);

/* 2009 年秋季 且 2010 年春季开设的课程 */
(select course_id from section where sem = 'Fall' and year = 2009)
intersect
(select course_id from section where sem = 'Spring' and year = 2010);

/* 2009 年秋季开设 但 2010 年春季未开设的课程 */
(select course_id from section where sem = 'Fall' and year = 2009)
except
(select course_id from section where sem = 'Spring' and year = 2010);
```

### 3.4 连接 (Join Expressions)

#### 自然连接 (Natural Join)

在两个关系的所有公共属性上自动进行等值连接。

```sql
select name, course_id
from student natural join takes;
```

#### 连接条件 (Join ... On ...)

```sql
select *
from student join takes on student.ID = takes.ID;
```

#### 外连接 (Outer Join)

保留在一个关系中存在但在另一个关系中没有匹配元组的元组。

  * **左外连接 (Left Outer Join)**
    保留左侧关系中没有匹配的元组。

    ```sql
    select *
    from course natural left outer join prereq;
    ```

  * **右外连接 (Right Outer Join)**
    保留右侧关系中没有匹配的元组。

    ```sql
    select *
    from course natural right outer join prereq;
    ```

  * **全外连接 (Full Outer Join)**
    保留两侧关系中没有匹配的元组。

    ```sql
    select *
    from course natural full outer join prereq;
    ```

### 3.5 嵌套子查询 (Nested Subqueries)

子查询是嵌套在另一个查询中的 `select-from-where` 表达式。

#### `IN` / `NOT IN` (集合成员)

```sql
/* 2009 年秋季 且 2010 年春季开设的课程 */
select distinct course_id
from section
where semester = 'Fall' and year = 2009 and
      course_id in (select course_id
                    from section
                    where semester = 'Spring' and year = 2010);

/* 2009 年秋季开设 但 2010 年春季未开设的课程 */
select distinct course_id
from section
where semester = 'Fall' and year = 2009 and
      course_id not in (select course_id
                        from section
                        where semester = 'Spring' and year = 2010);
```

#### `SOME` / `ALL` (集合比较)

  * `> some`：大于子查询结果中的“至少一个”。
    ```sql
    select name
    from instructor
    where salary > some (select salary
                         from instructor
                         where dept_name = 'Biology');
    ```
  * `> all`：大于子查询结果中的“所有”。
    ```sql
    select name
    from instructor
    where salary > all (select salary
                        from instructor
                        where dept_name = 'Biology');
    ```

#### `EXISTS` / `NOT EXISTS` (存在性测试)

测试子查询结果是否为空。`exists r` 当 r 非空时为 true。

```sql
/* 2009 年秋季 且 2010 年春季开设的课程 (相关子查询) */
select course_id
from section as S
where semester = 'Fall' and year = 2009 and
      exists (select *
              from section as T
              where semester = 'Spring' and year = 2010
                    and S.course_id = T.course_id);

/* 找出选修了 Biology 系所有课程的学生 */
select distinct S.ID, S.name
from student as S
where not exists ( (select course_id  /* Biology 系的所有课程 */
                    from course
                    where dept_name = 'Biology')
                   except  /* 减去 */
                   (select T.course_id  /* S 学生已选修的课程 */
                    from takes as T
                    where S.ID = T.ID) );
```

#### `FROM` 子句中的子查询

```sql
/* 找出平均工资大于 42000 的系及其平均工资 */
select dept_name, avg_salary
from (select dept_name, avg(salary) as avg_salary
      from instructor
      group by dept_name)
where avg_salary > 42000;
```

#### `WITH` 子句

定义一个临时的“视图”，仅在当前查询中可用。

```sql
/* 找出预算最高的系 */
with max_budget (value) as (
    select max(budget)
    from department
)
select department.name
from department, max_budget
where department.budget = max_budget.value;
```

#### `SELECT` 子句中的标量子查询

返回单个值的子查询。

```sql
/* 列出所有系及其讲师人数 */
select dept_name,
       (select count(*)
        from instructor
        where department.dept_name = instructor.dept_name)
       as num_instructors
from department;
```

### 3.6 视图 (Views)

视图是一种虚拟关系，它不存储数据，而是存储其定义。

```sql
create view faculty as
	select ID, name, dept_name
	from instructor;

create view departments_total_salary(dept_name, total_salary) as
	select dept_name, sum(salary)
	from instructor
	group by dept_name;
```

**可更新的视图**（即可以在视图上进行 `insert`, `delete`, `update`）必须满足特定条件：

  * `from` 子句中只有一个关系。
  * `select` 子句中只包含属性名（没有表达式、聚合或 `distinct`）。
  * 未在 `select` 中出现的属性必须可以为 `null`。
  * 查询中没有 `group by` 或 `having` 子句。

## 4\. DCL (数据控制语言)

### 4.1 授予权限 (GRANT)

用于授予用户或角色权限。

```sql
grant <privilege list>
on <relation or view name>
to <user list>;
```

  * **权限列表**：`select`, `insert`, `update`, `delete`, `all privileges`。
  * **用户列表**：用户 ID, `public` (所有用户) 或角色。

```sql
grant select on instructor to U1, U2, UL3;
```

### 4.2 撤销权限 (REVOKE)

用于撤销已授予的权限。

```sql
revoke <privilege list>
on <relation or view name>
from <user list>;
```

```sql
revoke select on branch from U1, U2, U3;
```