---
name: SPARQL Knowledge Graph Queries
description: For tasks requiring SPARQL queries against RDF/Turtle datasets, especially academic or organizational data.
---

## SPARQL Knowledge Graph Queries

### Workflow
1. **Read full data** - Use `head -n 500` or similar to get complete file. Count entities: `grep -c "a uni:Person"` to verify.
2. **Map schema** - List ALL predicates: `grep "uni:" file.ttl | grep -oE "uni:[a-zA-Z]+" | sort -u`
3. **Decompose criteria** - For EACH criterion, ask: "Can this be satisfied by a DIFFERENT entity than other criteria?" If yes, use separate FILTER EXISTS blocks. Example: "works in EU dept" + "some dept has >10 students" → these could be DIFFERENT departments!
4. **Build incrementally** - Start with most restrictive filter, add one constraint at a time
5. **Test with Python rdflib** - Always verify: `python3 -c "from rdflib import Graph; g=Graph(); g.parse('file.ttl'); print(list(g.query(open('query.sparql').read())))"`

### EU Member States (2025)
```
AT, BE, BG, CY, CZ, DE, DK, EE, ES, FI, FR, GR, HR, HU, IE, IT, LT, LU, LV, MT, NL, PL, PT, RO, SE, SI, SK
```

### Common Patterns

**Multi-condition filtering with aggregation:**
```sparql
SELECT ?name (GROUP_CONCAT(DISTINCT ?country; separator=", ") AS ?countries)
WHERE {
  ?prof a :Professor ;
        :hasRank :FullProfessor ;
        :name ?name ;
        :worksIn ?dept .
  ?dept :locatedIn ?country ;
        :hasEnrollment ?count .
  FILTER(?country IN ("AT", "BE", ...))
  FILTER(?count > 10)
}
GROUP BY ?name
```

**Enrollment via class relationship:**
```sparql
# If enrollment is on classes, not departments directly
?dept :offersClass ?class .
?class :enrollment ?count .
```

### Critical Checks
| Issue | Fix |
|-------|-----|
| Incomplete data read | Re-read without line limits; verify entity counts with grep |
| Wrong predicate path | Trace one entity manually through the graph |
| Aggregation mismatch | Ensure GROUP BY matches non-aggregated SELECT vars |
| Country code format | Check if data uses "GR" vs "Greece" vs URI |
| Numeric comparison | Verify `?count` is numeric, not string; use `xsd:integer` cast if needed |
| Correlated subquery | **AVOID** - `?var` in nested SELECT doesn't bind to outer. Use flat pattern with FILTER |

### Avoid These Anti-Patterns

**1. Correlated subquery scope issue:**
```sparql
# BAD: ?prof doesn't correlate into subquery
FILTER EXISTS {
  { SELECT ?dept WHERE { ?prof uni:worksIn ?dept } GROUP BY ?dept HAVING(COUNT(*)>10) }
}
```

**2. Combining independent criteria in same subquery:**
```sparql
# BAD: Requires SAME dept to be EU AND have >10 students
?prof uni:worksIn ?deptBig .
{ SELECT ?deptBig WHERE {
    ?deptBig uni:belongsTo ?univ . ?univ uni:locatedInCountry ?c .
    FILTER(?c IN ("ES","GR",...))  # EU check
    ...
    HAVING(COUNT(?student) > 10)   # enrollment check
  }
}

# GOOD: Separate FILTER EXISTS for independent criteria
?prof uni:worksIn ?euDept .
?euDept uni:belongsTo ?euUniv .
?euUniv uni:locatedInCountry ?euCountry .
FILTER(?euCountry IN ("AT","BE",...))

FILTER EXISTS {
  ?prof uni:worksIn ?anyDept .
  ?course uni:isTaughtIn ?anyDept .
  # ... count students in anyDept > 10
}
```

### Output Format
Save query to specified output file (e.g., `/app/solution.sparql`). Ensure no extra whitespace or comments break execution.
