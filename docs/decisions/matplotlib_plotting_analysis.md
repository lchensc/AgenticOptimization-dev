# Matplotlib Plotting Analysis

## Current Situation

**Terminal ASCII Plotting (asciichartpy):**
- ✅ Immediate display in terminal
- ✅ Works over SSH without GUI
- ✅ Lightweight and fast
- ❌ Limited to ~60-80 characters width
- ❌ Low resolution and visual quality
- ❌ Storage only keeps last 20 iterations (line 166 in `active_run.py`)
- ❌ No zoom, pan, or interaction
- ❌ Hard to show multiple metrics simultaneously

**Problem discovered:**
The storage intentionally limits iteration history to last 20 points for efficiency:
```python
# aopt/runs/active_run.py:166
"iterations": self.iterations[-20:] if len(self.iterations) > 20 else self.iterations,
```

This means downsampling isn't needed - plots will never exceed 20 iterations anyway.

## Matplotlib Integration Options

### Option 1: File Export (Recommended)

**Command syntax:**
```bash
paola> /plot 1 --export              # Save to default location
paola> /plot 1 --export png          # Specify format (png, pdf, svg)
paola> /plot 1 --export plots/run1.png  # Custom path
paola> /plot compare 1 2 3 --export  # Export comparison plot
```

**Implementation:**
```python
def handle_plot(self, run_id: int, export: Optional[str] = None):
    """Plot convergence history.

    Args:
        run_id: Run ID to plot
        export: Export to file (format or path). If None, show ASCII in terminal.
    """
    if export:
        self._plot_matplotlib(run_id, export)
    else:
        self._plot_ascii(run_id)  # Current implementation

def _plot_matplotlib(self, run_id: int, export: str):
    """Generate high-quality matplotlib plot."""
    import matplotlib.pyplot as plt

    run = self.storage.load_run(run_id)
    iterations = run.result_data.get('iterations', [])
    objectives = [it['objective'] for it in iterations]

    # Determine start iteration (if showing subset)
    total_evals = run.n_evaluations
    start_iter = max(0, total_evals - len(objectives))
    x_values = range(start_iter, start_iter + len(objectives))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_values, objectives, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective Value', fontsize=12)
    ax.set_title(f'Convergence History - Run #{run_id}: {run.algorithm} on {run.problem_name}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add info box
    info_text = f"Initial: {objectives[0]:.6e}\nFinal: {objectives[-1]:.6e}\nImprovement: {objectives[0]-objectives[-1]:.6e}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Handle showing subset
    if len(objectives) < total_evals:
        ax.set_title(f'{ax.get_title()}\n(Showing last {len(objectives)} of {total_evals} iterations)',
                     fontsize=12)

    # Determine output path
    if export in ['png', 'pdf', 'svg']:
        output_path = f'plots/run_{run_id}.{export}'
    else:
        output_path = export

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    self.console.print(f"\n[green]✓[/green] Plot saved to: [cyan]{output_path}[/cyan]\n")
```

**Pros:**
- ✅ High-quality publication-ready figures (300 DPI)
- ✅ Works over SSH (no GUI required)
- ✅ Can save in multiple formats (PNG, PDF, SVG)
- ✅ Preserves plots for later analysis
- ✅ Can batch export all runs
- ✅ Backward compatible (ASCII remains default)

**Cons:**
- ❌ Not immediately visible (user must open file)
- ❌ Extra step to view

**Directory structure:**
```
AgenticOptimization/
├── plots/               # Auto-created
│   ├── run_1.png
│   ├── run_2.png
│   ├── compare_1_2_3.png
│   └── session_2025-12-12/
│       └── run_4.png
```

---

### Option 2: Interactive Display

**Command syntax:**
```bash
paola> /plot 1 --show     # Display in matplotlib window
paola> /plot 1 --show --block  # Block CLI until window closed
```

**Implementation:**
```python
def _plot_matplotlib_show(self, run_id: int, block: bool = False):
    """Display interactive matplotlib plot."""
    import matplotlib.pyplot as plt

    # ... same plotting code as Option 1 ...

    if block:
        plt.show()  # Blocks until window closed
    else:
        plt.show(block=False)  # Non-blocking
        self.console.print("\n[yellow]Note: Plot window opened in background[/yellow]\n")
```

**Pros:**
- ✅ Immediate visual feedback
- ✅ Interactive (zoom, pan, save manually)
- ✅ Can keep window open while continuing CLI work

**Cons:**
- ❌ Requires GUI environment (X11 on Linux, doesn't work over SSH)
- ❌ Window management issues in some environments
- ❌ Plot disappears when program exits (unless saved manually)

---

### Option 3: Hybrid Approach (Recommended)

**Keep ASCII as default, add optional matplotlib export:**

```bash
# Quick preview in terminal (immediate)
paola> /plot 1
[ASCII chart displayed immediately]

# High-quality export when needed
paola> /plot 1 --export
✓ Plot saved to: plots/run_1.png

# Comparison still works in both modes
paola> /plot compare 1 2 3          # ASCII overlay
paola> /plot compare 1 2 3 --export # Matplotlib multi-line plot
```

**Implementation:**
- Keep existing ASCII code unchanged
- Add `--export` flag to both `/plot` and `/plot compare`
- Auto-create `plots/` directory
- Use consistent naming: `run_{id}.png`, `compare_{ids}.png`

**Pros:**
- ✅ Best of both worlds
- ✅ Fast feedback (ASCII) + high quality when needed (matplotlib)
- ✅ No breaking changes
- ✅ Works in all environments

**Cons:**
- ❌ Slightly more code to maintain

---

### Option 4: HTML Report Generation

**Command syntax:**
```bash
paola> /report              # Generate HTML report for current session
paola> /report --all        # Include all historical runs
paola> /report --runs 1 2 3 # Specific runs only
```

**Generates:**
```
reports/session_2025-12-12_143022.html
```

**HTML includes:**
- Embedded matplotlib plots (base64 encoded)
- Run comparison tables
- Statistical summary
- Links to raw data

**Pros:**
- ✅ Self-contained single file
- ✅ Easy to share and archive
- ✅ Professional presentation
- ✅ Can include additional analysis

**Cons:**
- ❌ Most complex to implement
- ❌ May be overkill for quick analysis

---

## Recommended Implementation Plan

### Phase 1: Add Matplotlib Export (1-2 hours)

1. **Add matplotlib to requirements.txt:**
   ```
   matplotlib>=3.7.0  # High-quality plotting
   ```

2. **Update `commands.py`:**
   - Add `_plot_matplotlib()` method for single plots
   - Add `_plot_matplotlib_compare()` for comparison plots
   - Modify `handle_plot()` to accept optional `export` parameter
   - Modify `handle_plot_compare()` to accept optional `export` parameter

3. **Update `repl.py` command parsing:**
   ```python
   elif cmd.startswith('/plot '):
       parts = cmd.split()
       if 'compare' in parts:
           run_ids = [int(p) for p in parts if p.isdigit()]
           export = parts[-1] if '--export' in cmd else None
           handler.handle_plot_compare(run_ids, export)
       else:
           run_id = int(parts[1])
           export = parts[-1] if '--export' in cmd else None
           handler.handle_plot(run_id, export)
   ```

4. **Add `.gitignore` entry:**
   ```
   plots/
   *.png
   *.pdf
   ```

### Phase 2: Enhanced Features (optional)

- Multi-panel plots (objective + gradient norm + constraint violation)
- Batch export all runs: `/export-all`
- Custom styling/themes
- Logarithmic scale option
- Confidence intervals for noisy objectives

---

## matplotlib Plot Features to Include

### Single Plot (`/plot 1 --export`):

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Main convergence line
ax.plot(x, objectives, 'b-', linewidth=2, marker='o', markersize=4, label='Objective')

# Formatting
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Objective Value', fontsize=12)
ax.set_title(f'Run #{run_id}: {algorithm} on {problem}', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best')

# Info box with stats
stats_text = f"""Initial: {obj[0]:.6e}
Final: {obj[-1]:.6e}
Improvement: {obj[0]-obj[-1]:.6e}
Evaluations: {len(obj)}"""
ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save high-res
fig.savefig(output_path, dpi=300, bbox_inches='tight')
```

### Comparison Plot (`/plot compare 1 2 3 --export`):

```python
fig, ax = plt.subplots(figsize=(12, 7))

colors = ['blue', 'red', 'green', 'orange', 'purple']
for i, run_data in enumerate(runs):
    ax.plot(run_data['x'], run_data['y'],
            color=colors[i], linewidth=2, marker='o', markersize=3,
            label=f"#{run_data['id']} ({run_data['algorithm']})")

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Objective Value', fontsize=12)
ax.set_title('Convergence Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)

fig.savefig(output_path, dpi=300, bbox_inches='tight')
```

---

## Storage Limitation Issue

**Current behavior (line 166 in `active_run.py`):**
```python
"iterations": self.iterations[-20:] if len(self.iterations) > 20 else self.iterations,
```

**Impact:**
- Runs with >20 iterations only store last 20
- Loses early exploration behavior
- Can't plot full convergence history

**Options:**

1. **Keep as-is** (storage efficiency)
   - Plots show "final convergence behavior"
   - Add note: "(Showing last 20 of 100 iterations)"

2. **Store all iterations** (comprehensive)
   - Change to: `"iterations": self.iterations`
   - Better for analysis, slightly larger storage

3. **Configurable limit** (flexible)
   - Add setting: `max_iterations_stored = 100`
   - Balance between storage and detail

**Recommendation:** Option 2 - storage is cheap, convergence analysis is valuable.

---

## Dependencies Required

Add to `requirements.txt`:
```
matplotlib>=3.7.0  # High-quality plotting and visualization
```

Optional for enhanced features:
```
seaborn>=0.12.0   # Statistical plotting themes (optional)
plotly>=5.14.0    # Interactive HTML plots (optional, for Phase 2)
```

---

## Example Usage Workflow

```bash
$ python paola_cli.py

paola> optimize 10D Rosenbrock with SLSQP
✓ Run #1 complete: objective=0.023456

paola> /plot 1
[ASCII chart displayed for quick feedback]

paola> /plot 1 --export
✓ Plot saved to: plots/run_1.png

paola> optimize same problem with BFGS
✓ Run #2 complete: objective=0.018234

paola> /plot compare 1 2
[ASCII overlay displayed]

paola> /plot compare 1 2 --export
✓ Plot saved to: plots/compare_1_2.png

paola> optimize same problem with L-BFGS-B
✓ Run #3 complete: objective=0.015678

paola> /plot compare 1 2 3 --export pdf
✓ Plot saved to: plots/compare_1_2_3.pdf
```

---

## Conclusion

**Recommended approach: Option 3 (Hybrid)**

- Keep ASCII plots as default for immediate feedback
- Add `--export` flag for matplotlib high-quality plots
- Simple implementation (~150 lines of code)
- Backward compatible
- Works in all environments

**Implementation priority:**
1. Phase 1: Basic matplotlib export (high priority, simple)
2. Fix storage to keep all iterations (medium priority, one-line change)
3. Phase 2: Enhanced features (low priority, nice-to-have)

**Benefits:**
- Professional publication-ready figures
- No terminal width limitations
- Preserves plots for documentation
- Interactive analysis capability
- Better for presentations and reports
