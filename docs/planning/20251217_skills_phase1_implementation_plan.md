# Paola Skills Phase 1 Implementation Plan

**Date**: 2025-12-17
**Goal**: Implement core Skills infrastructure

## Implementation Order

### Step 1: Directory Structure & Registry
Create the skills directory structure and registry.yaml.

```
paola/skills/
├── __init__.py              # Package init, exports
├── loader.py                # SkillLoader class
├── index.py                 # SkillIndex class
├── tools.py                 # @tool decorated functions
├── registry.yaml            # Global skill index
└── optimizers/              # Optimizer skills category
    └── ipopt/
        ├── skill.yaml       # Main skill definition
        ├── options.yaml     # Full option reference
        └── configurations.yaml  # Pre-built configs
```

### Step 2: SkillLoader Implementation
Core class for loading and parsing skill files.

```python
class SkillLoader:
    def load_metadata(skill_name) -> dict      # Level 1
    def load_overview(skill_name) -> str       # Level 2
    def load_section(skill_name, section) -> str  # Level 3
    def load_configuration(skill_name, config_name) -> dict
```

### Step 3: SkillIndex Implementation
Index for skill discovery and listing.

```python
class SkillIndex:
    def build_index() -> None
    def list_skills(category=None) -> list[dict]
    def get_skill_path(skill_name) -> Path
    def search(query, limit=3) -> list[dict]  # Basic keyword search first
```

### Step 4: Skill Tools
Four LangChain @tool decorated functions.

```python
@tool list_skills(category=None) -> str
@tool load_skill(skill_name, section="overview") -> str
@tool query_skills(query, limit=3) -> str
@tool get_skill_configuration(skill_name, config_name) -> str
```

### Step 5: IPOPT Skill Content
Create the first complete optimizer skill with:
- skill.yaml with Paola integration fields
- options.yaml with key IPOPT options
- configurations.yaml with common configs

### Step 6: Backend Passthrough
Update IPOPTBackend to pass all config options through (remove hardcoded mapping).

## File Dependencies

```
registry.yaml ──────────────────────────────────┐
                                                │
skill.yaml ─────┐                               │
options.yaml ───┼── SkillLoader ── SkillIndex ──┼── tools.py
configurations.yaml                             │
                                                │
__init__.py ────────────────────────────────────┘
```

## Testing Strategy

1. Unit test SkillLoader with IPOPT skill
2. Unit test SkillIndex listing and search
3. Integration test tools with mock agent
4. Manual test via Paola CLI

## Success Criteria

- [ ] `list_skills()` returns IPOPT with description
- [ ] `load_skill("ipopt")` returns overview
- [ ] `load_skill("ipopt", "options.warm_start")` returns warm-start options
- [ ] `get_skill_configuration("ipopt", "warm_start_from_parent")` returns JSON config
- [ ] Config from skill works with `run_optimization()`
