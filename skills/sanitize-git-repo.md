---
name: sanitize-git-repo
description: Guide for sanitizing git repositories by identifying and replacing sensitive information such as API keys, tokens, and credentials. This skill should be used when tasks involve removing secrets from codebases, sanitizing repositories before sharing, or replacing sensitive values with placeholders. Applies to tasks involving secret detection, credential removal, or repository cleanup for security purposes.
---

# Sanitize Git Repository

This skill provides guidance for systematically identifying and replacing sensitive information in git repositories, including API keys, tokens, passwords, and other credentials.

## ⚠️ Critical: Follow Discovery Order

**Common failure mode:** Agents skip specific pattern searches and only do generic searches (like "token"), missing GitHub tokens (`ghp_`), HuggingFace tokens (`hf_`), and secrets embedded in JSON files.

**MANDATORY:** You MUST complete Phase 1 discovery in the exact order specified:
1. Search for specific prefixes FIRST (`ghp_`, `hf_`, `AKIA`)
2. Check JSON files explicitly (they contain embedded secrets)
3. Only then proceed to replacement

Do NOT skip to replacement after finding only one type of secret (e.g., AWS keys). You must find ALL types before replacing.

## When to Use This Skill

- Sanitizing a repository before sharing or open-sourcing
- Removing accidentally committed secrets from a codebase
- Replacing hardcoded credentials with placeholders
- Auditing a repository for sensitive information
- Preparing code for security review

## Recommended Approach

### Phase 1: Comprehensive Discovery (MANDATORY ORDER)

**CRITICAL: Complete ALL discovery steps below in order BEFORE making any replacements.** Build a complete inventory of all sensitive values before making any changes. This prevents the common mistake of discovering additional secrets mid-process.

**Step 1: Search for Specific Token Prefixes (DO THIS FIRST)**

Search for specific token patterns using exact prefixes. These are the most reliable indicators:

```bash
# GitHub tokens - search for ghp_ prefix
grep -rn "ghp_" . --exclude-dir=.git

# HuggingFace tokens - search for hf_ prefix  
grep -rn "hf_" . --exclude-dir=.git

# AWS Access Keys - search for AKIA prefix
grep -rn "AKIA[A-Z0-9]\{16\}" . --exclude-dir=.git
```

**Why first?** These specific patterns have low false-positive rates and are easy to identify. Generic searches (like "token") produce too many false positives.

**Step 2: Search for AWS Secret Keys**

```bash
# AWS Secret Access Keys (40-char strings, often near AWS_ACCESS_KEY_ID)
grep -rn "AWS_SECRET_ACCESS_KEY" . --exclude-dir=.git
grep -rn "secret_access_key" . --exclude-dir=.git -i
```

**Step 3: Explicitly Check JSON Files**

**MANDATORY:** JSON files often contain embedded secrets (e.g., git diff content stored as JSON strings). You MUST check them separately:

```bash
# Find all JSON files
find . -name "*.json" -not -path "./.git/*"

# Search for token patterns in JSON files specifically
grep -rn "ghp_\|hf_\|AKIA" . --include="*.json" --exclude-dir=.git
```

**Step 4: Check Configuration Files**

```bash
# YAML/YML files
grep -rn "ghp_\|hf_\|AKIA\|AWS_SECRET" . --include="*.yaml" --include="*.yml" --exclude-dir=.git

# Python files
grep -rn "ghp_\|hf_\|AKIA\|AWS_SECRET" . --include="*.py" --exclude-dir=.git
```

**Step 5: Verify Discovery Completeness**

Before proceeding to replacement, confirm you've found:
- [ ] All `ghp_` patterns (GitHub tokens)
- [ ] All `hf_` patterns (HuggingFace tokens)  
- [ ] All `AKIA` patterns (AWS access keys)
- [ ] All AWS secret keys (40-char strings)
- [ ] All JSON files have been checked
- [ ] All configuration files (YAML, Python) have been checked

**Common Secret Patterns Reference:**

| Type | Pattern/Prefix | Example | Notes |
|------|----------------|---------|-------|
| AWS Access Keys | `AKIA[A-Z0-9]{16}` | `AKIAIOSFODNN7EXAMPLE` | Exactly 16 chars after AKIA |
| AWS Secret Keys | 40-char base64 strings | `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` | Mixed case, often near access keys |
| GitHub Tokens | `ghp_`, `gho_`, `ghs_`, `ghr_` | `ghp_xxxxxxxxxxxx` | `ghp_` followed by 36 alphanumeric chars |
| Huggingface Tokens | `hf_` | `hf_xxxxxxxxxxxx` | **Varies in length**: 20-40 chars after prefix |

**Critical: Token Format Variations**

- **HuggingFace tokens vary in length!** Use flexible pattern: `hf_[a-zA-Z0-9]{20,40}` (not fixed length)
- **AWS Secret Keys have mixed case!** Must match case-insensitively or use context-based matching
- **Multiple tokens of same type may exist** - all must be replaced
- **JSON files are a trap!** Secrets appear embedded in JSON string values with escape characters - check them explicitly

### Phase 2: Inventory Documentation

**DO NOT proceed to replacement until you have completed Phase 1 discovery.**

Before making changes, create a documented list of:

1. Each unique sensitive value found (from Phase 1)
2. All file locations where each value appears
3. The exact string to match (including surrounding context if needed for uniqueness)
4. The placeholder to use for replacement

**Verify your inventory includes:**
- All GitHub tokens (`ghp_*`) found in Step 1
- All HuggingFace tokens (`hf_*`) found in Step 1
- All AWS keys (`AKIA*` and secret keys) found in Steps 1-2
- All secrets found in JSON files (Step 3)
- All secrets found in configuration files (Step 4)

**Placeholder Conventions:**

- Use descriptive placeholders that indicate the type: `<AWS_ACCESS_KEY>`, `<GITHUB_TOKEN>`, `<DATABASE_PASSWORD>`
- Maintain consistency across all replacements
- Match the format specified by the user if provided

### Phase 3: Systematic Replacement

Execute replacements methodically:

1. Work through the inventory one secret at a time
2. For each secret, replace ALL occurrences across ALL files
3. Verify each replacement immediately after making it
4. Use exact string matching to avoid unintended modifications

**Replacement Best Practices:**

- Read files before editing to ensure exact string matching
- Handle whitespace and formatting precisely
- Consider secrets embedded in JSON or other structured formats (may require escaping)
- Use batch replacements when the same value appears multiple times in one file
- **DO NOT use sed for complex files** - Use Python for reliable regex-based replacement
- **DO NOT parse JSON** - treat as plain text for replacement to avoid breaking structure

**Python Replacement Example:**

For reliable replacement across multiple files, use Python with regex:

```python
import re

def sanitize_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # AWS Access Key ID (AKIA followed by 16 alphanumeric chars)
    content = re.sub(r'AKIA[A-Z0-9]{16}', '<your-aws-access-key-id>', content)

    # AWS Secret Access Key - context-based matching (case-insensitive)
    content = re.sub(
        r'(AWS_SECRET_ACCESS_KEY[=:]\s*["\']?)([A-Za-z0-9/+=]{40})',
        r'\1<your-aws-secret-access-key>',
        content,
        flags=re.IGNORECASE
    )
    content = re.sub(
        r'(secret_access_key[=:]\s*["\']?)([A-Za-z0-9/+=]{40})',
        r'\1<your-aws-secret-access-key>',
        content,
        flags=re.IGNORECASE
    )

    # GitHub tokens (ghp_ followed by 36 alphanumeric chars)
    content = re.sub(r'ghp_[a-zA-Z0-9]{36}', '<your-github-token>', content)

    # HuggingFace tokens - FLEXIBLE LENGTH (20-40 chars after hf_)
    content = re.sub(r'hf_[a-zA-Z0-9]{20,40}', '<your-huggingface-token>', content)

    with open(filepath, 'w') as f:
        f.write(content)
```

**The JSON File Trap:**

JSON files may contain **embedded YAML with git diff content** or other structured data where secrets appear:
1. Inside JSON string values
2. With escape characters (`\n`, etc.)
3. Multiple times per secret

**You MUST search JSON files separately and replace ALL occurrences.** Treat JSON as plain text for replacement, not as structured data.

### Phase 4: Verification

After all replacements:

1. Re-run all original discovery searches to confirm no secrets remain
2. Search for partial matches of sensitive values
3. Run any provided test suites to validate the sanitization
4. Check that placeholders are properly formatted

## Common Pitfalls

### 1. Skipping Specific Pattern Searches (MOST COMMON FAILURE)

**Problem:** Agent searches for generic terms like "token" instead of specific prefixes (`ghp_`, `hf_`), resulting in thousands of false positives and missing actual secrets. Agent then declares task complete after finding only one type of secret (e.g., AWS keys).

**Prevention:** 
- **ALWAYS search for specific prefixes FIRST:** `ghp_`, `hf_`, `AKIA`
- **DO NOT** do generic searches like `grep -r "token"` - this produces too many false positives
- **DO NOT** stop after finding only one type of secret - you must find ALL types before replacing
- Follow the exact discovery order in Phase 1

### 2. Missing JSON Files

**Problem:** Secrets appear in JSON files (often embedded as git diff content in JSON strings), but agent doesn't check JSON files explicitly.

**Prevention:** 
- **MANDATORY Step 3:** Explicitly find and search all JSON files
- Use: `find . -name "*.json"` then `grep -rn "ghp_\|hf_\|AKIA" . --include="*.json"`
- JSON files are a known trap - they must be checked separately

### 3. Incomplete Initial Discovery

**Problem:** Secrets discovered incrementally during the replacement process, requiring backtracking.

**Prevention:** Invest time upfront in comprehensive searching using multiple patterns and checking all file types, including data files and embedded content. Complete ALL discovery steps before starting replacements.

### 2. Secrets in Unexpected Locations

**Problem:** Secrets appear in JSON files, embedded strings, test fixtures, or encoded data that aren't found by simple searches.

**Prevention:** Search recursively through all file types. Check JSON and YAML files specifically. Look for base64-encoded content that might contain secrets. **JSON files may contain embedded YAML with git diff content** - secrets can appear multiple times with escape characters.

### 3. Exact String Matching Failures

**Problem:** Edit operations fail due to whitespace differences or character encoding issues.

**Prevention:** Always read the target file first to understand exact formatting. Copy strings exactly as they appear in the file, including whitespace.

### 4. Missing Occurrences

**Problem:** Multiple occurrences of the same secret exist, but only some are replaced.

**Prevention:** After each replacement, immediately verify by searching for the original value again. Use replace-all functionality when available. Use regex `re.sub()` with global replacement (default behavior) to replace all occurrences in one pass.

### 7. Token Length Assumptions

**Problem:** Assuming fixed token lengths (e.g., expecting HuggingFace tokens to be exactly 34 chars) causes missed replacements.

**Prevention:** Use flexible regex patterns. For example, `hf_[a-zA-Z0-9]{20,40}` instead of `hf_[a-zA-Z0-9]{34}`.

### 8. Case Sensitivity Issues

**Problem:** AWS secret keys have mixed case, but regex matching fails due to case sensitivity.

**Prevention:** Use `re.IGNORECASE` flag or context-based matching for case-sensitive patterns.

### 5. Git History Considerations

**Problem:** Secrets remain in git history even after being removed from current files.

**Prevention:** Clarify with the user whether git history sanitization is required. If so, tools like `git filter-branch` or `BFG Repo-Cleaner` may be needed (this is a separate, more complex operation).

### 6. Tool Availability Assumptions

**Problem:** Specialized search tools (like ripgrep) may not be available in all environments.

**Prevention:** Have fallback approaches ready. Standard `grep -r` works in most environments. Test tool availability before relying on it.

## Mandatory Discovery Checklist

**Complete this checklist BEFORE starting replacements:**

- [ ] **Step 1 Complete:** Searched for `ghp_` (GitHub tokens)
- [ ] **Step 1 Complete:** Searched for `hf_` (HuggingFace tokens)
- [ ] **Step 1 Complete:** Searched for `AKIA` (AWS access keys)
- [ ] **Step 2 Complete:** Searched for AWS secret keys
- [ ] **Step 3 Complete:** Found and checked all JSON files explicitly
- [ ] **Step 4 Complete:** Checked all YAML/Python configuration files
- [ ] **Inventory Complete:** Documented all secrets found with file locations

## Verification Checklist

Before considering the task complete:

- [ ] All known secret patterns have been searched for (Discovery Phase complete)
- [ ] Configuration files have been examined
- [ ] JSON/YAML data files have been checked
- [ ] Each identified secret has been replaced in all locations
- [ ] Verification searches confirm no original secrets remain
- [ ] Placeholders follow a consistent format
- [ ] Any provided tests pass successfully

## Search Commands Reference

Standard grep (widely available):
```bash
# Search for AWS access keys
grep -rn "AKIA[A-Z0-9]\{16\}" . --exclude-dir=.git

# Search for common token prefixes (use flexible patterns)
grep -rn "ghp_\|gho_\|ghs_\|hf_" . --exclude-dir=.git

# Search for HuggingFace tokens (flexible length: 20-40 chars)
grep -rnE "hf_[a-zA-Z0-9]{20,40}" . --exclude-dir=.git

# Search for password-related strings
grep -rn -i "password\|passwd\|pwd" . --exclude-dir=.git

# Search for API key patterns
grep -rn -i "api[_-]\?key\|apikey" . --exclude-dir=.git

# Count occurrences to ensure you replace ALL of them
grep -ro "hf_[a-zA-Z0-9]\{20,40\}" . | wc -l
```

Always exclude the `.git` directory from searches to avoid noise from git internals.

## Verification Commands

After replacement, verify NO secrets remain:

```bash
# Must return empty/no matches - use flexible patterns!
grep -iE "(AKIA[A-Z0-9]{16}|ghp_[a-zA-Z0-9]{36}|hf_[a-zA-Z0-9]{20,})" \
    <target-files>

# Check for AWS secret keys by context (40-char strings after key patterns)
grep -iE "(secret_access_key|aws_secret)[=:].{0,5}[A-Za-z0-9/+=]{40}" \
    <target-files>

# Verify placeholders are present
grep "<your-" <target-files>
```

## Critical Workflow Reminders

**DO NOT:**
- Modify files outside the target list
- Use `git filter-branch` unless explicitly requested (only sanitize current files)
- Change file structure or formatting beyond secret replacement
- Parse JSON - treat as plain text for replacement
- Add new lines or code (e.g., no adding `mkdir ~/.cache/huggingface`)
- "Improve" the code - ONLY replace secret values with placeholders

**CRITICAL: The only change should be `secret_value` → `<placeholder>`. No other changes.**