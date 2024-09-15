
select * from table;

select count(*) from table2;

--generate series

select generate_series(1,10,2), col2
from table1



-- Partition by

select *, 
sum(sales) over (Partition by department) as total_sales
from table1

-- department	sales	total_sales
-- A	100	250
-- A	150	250
-- B	200	500
-- B	300	500
-- C	400	400



-- Order by

SELECT *, 
SUM(sales) OVER (PARTITION BY department ORDER BY date) AS cumulative_sales
FROM table1;

-- department	sales	date	cumulative_sales
-- A	    100	    2024-01-01	100
-- A	    150	    2024-01-02	250
-- B	    200	    2024-01-01	200
-- B	    300	    2024-01-02	500
-- C	    400	    2024-01-01	400



-- Rank 
SELECT department,
       sales,
       date,
       RANK() OVER (PARTITION BY department ORDER BY sales DESC) AS sales_rank
FROM table1;


-- moving average
    
    -- rows BETWEEN PRECEDING

    SELECT date,
        sales_amount,
        AVG(sales_amount) OVER (ORDER BY date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg
    FROM sales;



    -- Cumulative Sum

    SELECT date,
       sales_amount,
       SUM(sales_amount) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_sales
    FROM sales;

-- Lead/ lag
    --syntax
SELECT date,
       sales_amount,
       LEAD(sales_amount, 1) OVER (ORDER BY date) AS next_day_sales
FROM sales;

select date,
sale,
lead(sale,1,0) over (order by date)

    --Calculating Differences Between Rows:
    SELECT date,
       sales_amount,
       sales_amount - LAG(sales_amount) OVER (ORDER BY date) AS daily_change
    FROM sales;


