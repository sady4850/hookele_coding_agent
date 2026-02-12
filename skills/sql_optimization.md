---
name: SQL Query Optimization
description: For tasks requiring SQL query speedups, database tuning, or fixing slow queries.
---

## SQL Query Optimization

### Strategy
1. **Analyze first:** Run `EXPLAIN ANALYZE` to identify bottlenecks (seq scans, nested loops, high row estimates)
2. **Reduce working set early:** Filter rows BEFORE expensive operations (joins, aggregations, window functions)

### Anti-Patterns to Fix
| Problem | Solution |
|---------|----------|
| Correlated subquery `(SELECT x FROM t WHERE t.id = outer.id)` | Window function or JOIN |
| `SELECT *` with large tables | Select only needed columns |
| Window function on full table | Filter in CTE first, then apply window |
| Multiple scans of same table | Single CTE with all needed aggregations |

### Window Function Pattern
```sql
-- BAD: Window over entire table
SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY val) FROM huge_table;

-- GOOD: Filter first, then window
WITH filtered AS (
  SELECT * FROM huge_table WHERE some_condition
)
SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY val) FROM filtered;
```

### Bottom-Up Aggregation
When you need both aggregates AND "top item per group":
```sql
WITH granular AS (
  -- One scan: compute stats at finest grain
  SELECT word_id, synset_id, COUNT(*) as cnt FROM items GROUP BY 1, 2
),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY word_id ORDER BY cnt DESC) as rn FROM granular
)
SELECT word_id, SUM(cnt) as total, MAX(CASE WHEN rn=1 THEN synset_id END) as top_synset
FROM ranked GROUP BY word_id;
```

### Verification
- If baseline is too slow (>120s), verify on subset: `WHERE id BETWEEN 1 AND 500`
- Compare output row-by-row: optimized must match baseline exactly
- If optimized is slower, revert and try simpler approach
