# PAOLA Evaluator Registration Testing Guide

This guide walks you through testing the evaluator registration system in the PAOLA CLI.

## Prerequisites

1. **Installation**: Ensure PAOLA is installed
   ```bash
   cd /home/longchen/PythonCode/gendesign/AgenticOptimization
   ```

2. **Evaluator File**: We've created `evaluators.py` with 5 benchmark functions:
   - `rosenbrock` - Classic banana function (2D+)
   - `sphere` - Sum of squares (any dimension)
   - `rastrigin` - Highly multimodal (any dimension)
   - `ackley` - Multimodal with ridges (any dimension)
   - `beale` - 2D test function

3. **Verify evaluators work**:
   ```bash
   python evaluators.py
   ```
   Should show all functions evaluate correctly at their optima.

## Step-by-Step CLI Testing

### Step 1: Launch PAOLA CLI

```bash
python -m paola.cli
```

You should see:
```
╭─────────────────────────────────────────────────╮
│ PAOLA - Platform for Agentic Optimization with │
│            Learning and Analysis                │
│                                                 │
│ The optimization platform that learns from      │
│ every run                                       │
│                                                 │
│ Version 0.1.0                                   │
│                                                 │
│ Type your optimization goals in natural         │
│ language.                                       │
│ Type '/help' for commands, '/exit' to quit.    │
╰─────────────────────────────────────────────────╯

Initializing agent...
✓ Agent ready!

paola>
```

### Step 2: Check Available Commands

```bash
paola> /help
```

Look for the "Evaluator Registration" section:
```
Evaluator Registration:
  /register <file.py>        - Register an evaluator function
  /evaluators                - List all registered evaluators
  /evaluator <id>            - Show detailed evaluator configuration
```

### Step 3: List Existing Evaluators (Should be Empty)

```bash
paola> /evaluators
```

Expected output:
```
No evaluators registered yet

Use /register <file.py> to register an evaluator
```

### Step 4: Register Rosenbrock Function

```bash
paola> /register evaluators.py
```

**Interactive Prompts**:

1. **File contents displayed**:
   ```
   Registering evaluator from: evaluators.py
   Reading file...

   File contents:
   ╭────────────────────────────────────────────────╮
   │ """                                            │
   │ External evaluator functions for PAOLA...     │
   │ """                                            │
   │                                                │
   │ import numpy as np                             │
   │                                                │
   │ def rosenbrock(x):                             │
   │     """Rosenbrock function..."""              │
   │     ...                                        │
   ╰────────────────────────────────────────────────╯
   ```

2. **Enter function name**:
   ```
   Please provide the following information:

     Function name: rosenbrock
   ```

3. **Enter evaluator name** (press Enter for default):
   ```
     Evaluator name (default: same as function):
   ```
   (Just press Enter to accept default: "rosenbrock")

4. **Enter evaluator ID** (press Enter for default):
   ```
     Evaluator ID (default: rosenbrock_eval):
   ```
   (Just press Enter to accept default: "rosenbrock_eval")

5. **Configuration testing**:
   ```
   Testing configuration...
   ```
   The system will test the configuration by creating a FoundryEvaluator
   and running a test evaluation.

6. **Success**:
   ```
   Storing in Foundry...

   ╭────────────────────────────────────────────────╮
   │ ✓ Evaluator Registered Successfully            │
   │                                                │
   │ Evaluator ID:  rosenbrock_eval                 │
   │ Name:          rosenbrock                      │
   │ Source:        /path/to/evaluators.py          │
   │ Function:      rosenbrock                      │
   │                                                │
   │ You can now use this evaluator in              │
   │ optimizations                                  │
   ╰────────────────────────────────────────────────╯
   ```

### Step 5: Register More Evaluators

Repeat Step 4 for other functions:

```bash
paola> /register evaluators.py
  Function name: sphere
  Evaluator name (default: same as function):
  Evaluator ID (default: sphere_eval):

✓ Evaluator Registered Successfully
```

```bash
paola> /register evaluators.py
  Function name: rastrigin
  Evaluator name (default: same as function):
  Evaluator ID (default: rastrigin_eval):

✓ Evaluator Registered Successfully
```

### Step 6: List All Registered Evaluators

```bash
paola> /evaluators
```

Expected output:
```
                 Registered Evaluators
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ ID              ┃ Name       ┃ Type            ┃ Status ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ rosenbrock_eval │ rosenbrock │ python_function │ ●      │
│ sphere_eval     │ sphere     │ python_function │ ●      │
│ rastrigin_eval  │ rastrigin  │ python_function │ ●      │
└─────────────────┴────────────┴─────────────────┴────────┘

Total: 3 evaluators
```

### Step 7: Show Evaluator Details

```bash
paola> /evaluator rosenbrock_eval
```

Expected output:
```
╭────────────────────────────────────────────────────────╮
│ Evaluator: rosenbrock                                  │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│                                                        │
│ ID:           rosenbrock_eval                          │
│ Status:       registered                               │
│ Type:         python_function                          │
│ Source:       /path/to/evaluators.py                   │
│ Function:     rosenbrock                               │
│                                                        │
│ Capabilities:                                          │
│   • Observation gates: True                            │
│   • Caching: True                                      │
│                                                        │
│ Performance:                                           │
│   • Cost per eval: 1.0                                 │
│   • Total evaluations: 0                               │
│   • Success rate: N/A                                  │
╰────────────────────────────────────────────────────────╯
```

### Step 8: Use Registered Evaluator in Optimization (Agent-Driven)

Now you can use natural language to run optimizations with your registered evaluator:

```bash
paola> Optimize the rosenbrock function in 2 dimensions using SLSQP
```

The agent will:
1. Create the problem (2D Rosenbrock)
2. Start an optimization run
3. Use the `rosenbrock_eval` evaluator
4. Run SLSQP optimization
5. Report results

Or try:
```bash
paola> Compare SLSQP and BFGS on the sphere function in 5 dimensions
```

### Step 9: Verify Storage

Check that evaluators are persisted:

```bash
ls -la .paola_data/evaluators/
```

You should see JSON files:
```
rosenbrock_eval.json
sphere_eval.json
rastrigin_eval.json
```

Examine a config file:
```bash
cat .paola_data/evaluators/rosenbrock_eval.json
```

### Step 10: Exit CLI

```bash
paola> /exit
```

Or press `Ctrl+D`.

## Testing Checklist

Use this checklist to verify all features:

### Basic Registration
- [ ] `/register` command works
- [ ] File contents are displayed
- [ ] Interactive prompts work (function name, evaluator name, ID)
- [ ] Configuration testing executes successfully
- [ ] Success message is displayed
- [ ] Evaluator is stored in `.paola_data/evaluators/`

### Listing & Inspection
- [ ] `/evaluators` shows empty state initially
- [ ] `/evaluators` displays table with registered evaluators
- [ ] `/evaluator <id>` shows detailed configuration
- [ ] Status indicators work (color-coded)
- [ ] Total count is accurate

### Error Handling
- [ ] `/register nonexistent.py` shows error
- [ ] `/register file.txt` rejects non-.py files
- [ ] Invalid function name shows error
- [ ] `/evaluator invalid_id` shows error

### Multiple Evaluators
- [ ] Can register multiple functions from same file
- [ ] Each evaluator has unique ID
- [ ] All evaluators appear in `/evaluators` list
- [ ] Can switch between evaluators in optimization

### Agent Integration
- [ ] Agent can use registered evaluators in natural language
- [ ] Agent suggests registered evaluators when appropriate
- [ ] Optimization runs use registered evaluators correctly

## Troubleshooting

### Issue: "File not found"
**Solution**: Use absolute or relative path from AgenticOptimization directory
```bash
paola> /register evaluators.py           # ✓ Correct (in root)
paola> /register ./evaluators.py         # ✓ Also works
paola> /register /full/path/to/evaluators.py  # ✓ Absolute path
```

### Issue: "Configuration test failed"
**Solution**: Check that:
1. Function accepts numpy array or list as input
2. Function returns a scalar (float/int)
3. No import errors in the evaluator file
4. Function doesn't crash on test input `[1.0, 1.0]`

### Issue: "Evaluator not found"
**Solution**: Check spelling of evaluator_id
```bash
paola> /evaluators          # List all IDs
paola> /evaluator <id>      # Use exact ID from list
```

### Issue: CLI doesn't start
**Solution**:
```bash
# Ensure you're in the right directory
cd /home/longchen/PythonCode/gendesign/AgenticOptimization

# Try running directly
python -m paola.cli

# Or check Python path
python -c "import paola; print(paola.__file__)"
```

## Advanced Testing

### Test with Custom Evaluator

Create your own evaluator:

```python
# my_evaluator.py
import numpy as np

def custom_objective(x):
    """My custom optimization function."""
    x = np.atleast_1d(x)
    # Your custom logic here
    return float(np.sum((x - 2.0)**2))  # Minimum at x = [2, 2, ...]
```

Register it:
```bash
paola> /register my_evaluator.py
  Function name: custom_objective
```

### Test Agent-Driven Registration

Instead of using `/register`, try natural language:

```bash
paola> Register the rosenbrock function from evaluators.py
```

The agent should:
1. Call `read_file("evaluators.py")`
2. Analyze the code
3. Generate configuration for rosenbrock
4. Call `foundry_store_evaluator(...)`
5. Confirm success

### Test Performance Tracking

After running optimizations with a registered evaluator:

```bash
paola> /evaluator rosenbrock_eval
```

Check that performance metrics updated:
- Total evaluations > 0
- Success rate shows actual rate
- Cost per eval tracked

## Expected Behavior Summary

| Command | Expected Result |
|---------|----------------|
| `/register evaluators.py` | Interactive registration flow |
| `/evaluators` | Table of all evaluators or empty state |
| `/evaluator <id>` | Detailed configuration panel |
| `/help` | Shows registration commands |
| Natural language | Agent can register/use evaluators |

## Next Steps After Testing

1. **Production Use**: Register your actual engineering evaluators
2. **Optimization Runs**: Use registered evaluators in real optimizations
3. **Documentation**: Document your evaluator configurations
4. **Templates**: Create templates for common evaluator types
5. **Batch Registration**: Request feature for registering multiple evaluators at once

## Support

If you encounter issues:
1. Check `.paola_data/evaluators/` for config files
2. Review `evaluators.py` for function correctness
3. Check Python syntax with `python evaluators.py`
4. Verify imports: `python -c "from paola.foundry import FoundryEvaluator"`

Happy testing! 🚀
