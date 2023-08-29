
--- New column to amend average rating:
ALTER TABLE dbo.books ADD Avg_rtg AS CAST(average_rating/100 AS decimal (7,2))

---Delete new Column
--ALTER TABLE dbo.books
--DROP COLUMN Avg_rtg;

---Check new column Avg_rtg
SELECT TOP 10
title,
Avg_rtg
FROM dbo.books
order by Avg_rtg desc


--- Longest Books

SELECT TOP 10
title,
authors, 
num_pages
FROM dbo.books
ORDER BY num_pages DESC


--- Count Books by publisher
SELECT TOP 10
publisher,
COUNT (title) as total_books

FROM dbo.books
GROUP BY publisher
ORDER BY total_books DESC

--- Check books by year, extract Year
SELECT 
	YEAR(publication_date),
	COUNT (title) AS Num_Books

FROM dbo.books

GROUP BY YEAR(publication_date)

ORDER BY YEAR(publication_date) ASC


--- Let's check some information for the db:
exec sp_help 'dbo.books'


--- Avg Rtgs by Language
SELECT 
	language_code,
	ROUND (AVG(Avg_rtg),2) AS Average_Rtg_Language
	
FROM dbo.books
WHERE language_code is not null and Avg_rtg is not null	
GROUP BY  language_code
ORDER BY language_code


--- Books by Stephen King, using window functions to group repeated titles, getting their avg rtg and pages

SELECT
DISTINCT title,
authors,
AVG(Avg_rtg) OVER
         (PARTITION BY title) AS King_rtg,
AVG(num_pages) OVER
         (PARTITION BY title) AS King_pages

FROM dbo.books
WHERE authors LIKE 'Stephen King'



--- Eng Books
WITH cte2 AS 

(SELECT 
title,
CASE 
WHEN language_code ='eng'
	OR language_code  ='en-US'
	OR language_code  ='en-GB'
	OR language_code  ='en-CA'
			
THEN 'English'

		
END AS Eng_Lang

FROM dbo.books)

SELECT Eng_Lang,
COUNT (title) AS total_books_eng
FROM cte2
WHERE Eng_Lang is not null	
GROUP BY Eng_Lang


--- Avg Rtg & Pages per year
SELECT
DISTINCT YEAR(publication_date) as Year,

AVG(Avg_rtg) OVER
         (PARTITION BY YEAR(publication_date)) AS Year_rtg,
AVG(num_pages) OVER
         (PARTITION BY YEAR(publication_date)) AS Year_pages

FROM dbo.books
WHERE YEAR(publication_date) is not null
ORDER BY Year


