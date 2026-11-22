-- Count host transfers in a whole-graph Nsight Systems SQLite export.
.headers on
.mode column

SELECT count(*) AS graph_executions
FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE;

WITH graph_window AS (
    SELECT min(start) AS first_start, max(end) AS last_end
    FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE
), host_copies AS (
    SELECT direction.label AS direction, copy.bytes, coalesce(copy.copyCount, 1) AS copies,
           CASE
               WHEN graph_window.first_start IS NULL THEN 'no graph observed'
               WHEN copy.end <= graph_window.first_start THEN 'before first graph'
               WHEN copy.start >= graph_window.last_end THEN 'after last graph'
               WHEN EXISTS (
                   SELECT 1 FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE AS graph
                   WHERE copy.start < graph.end AND graph.start < copy.end
               ) THEN 'overlaps graph'
               ELSE 'between graphs'
           END AS phase
    FROM CUPTI_ACTIVITY_KIND_MEMCPY AS copy
    JOIN ENUM_CUDA_MEMCPY_OPER AS direction ON direction.id = copy.copyKind
    CROSS JOIN graph_window
    WHERE direction.label IN ('Host-to-Device', 'Device-to-Host')
)
SELECT direction, phase, sum(copies) AS copies, sum(bytes) AS bytes
FROM host_copies
GROUP BY direction, phase
ORDER BY direction, phase;
