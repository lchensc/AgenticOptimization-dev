# Quick Testing Steps

**Fast track to test evaluator registration in PAOLA CLI**

## 1. Start CLI
```bash
python -m paola.cli
```

## 2. Check help
```bash
paola> /help
```
Look for "Evaluator Registration" section.

## 3. List evaluators (should be empty)
```bash
paola> /evaluators
```

## 4. Register Rosenbrock
```bash
paola> /register evaluators.py
```

When prompted:
- Function name: `rosenbrock`
- Evaluator name: (press Enter for default)
- Evaluator ID: (press Enter for default)

Wait for "✓ Evaluator Registered Successfully"

## 5. Verify registration
```bash
paola> /evaluators
```

Should show table with `rosenbrock_eval`.

## 6. Show details
```bash
paola> /evaluator rosenbrock_eval
```

Should show configuration panel.

## 7. Register more (optional)
```bash
paola> /register evaluators.py
```
Try: `sphere`, `rastrigin`, `ackley`, `beale`

## 8. Use in optimization (agent-driven)
```bash
paola> Optimize the rosenbrock function in 2 dimensions using SLSQP
```

## 9. Exit
```bash
paola> /exit
```

## Verify Persistence
```bash
ls .paola_data/evaluators/
cat .paola_data/evaluators/rosenbrock_eval.json
```

---

**That's it!** See `TESTING_GUIDE.md` for comprehensive testing.
