---
name: Specification Accuracy
description: For tasks involving Protobuf, OpenAPI, or generating code from specifications.
---

## Specification accuracy
- Pay close attention to exact field names, parameter names, and types in specifications. If a spec says "a value (int)", check whether it means the field should be named "value" or just describes the type.
- For protocol definitions (protobuf, OpenAPI, etc.), match field names exactly as specified or as expected by tests. When in doubt, check test files or create a minimal test client to verify field names.
- After generating code from specs (proto files, schemas, etc.), verify the generated code matches test expectations by running tests or creating a simple client that exercises the API.
