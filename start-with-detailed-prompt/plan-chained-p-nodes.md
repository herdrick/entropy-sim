# Plan: Chained P-Node Distributions

## Overview

Refactor `foo.py` so that the app starts with only the Events section (rug plot + event controls). The user can then create a chain of P distribution nodes, one at a time, where each node either passes events through unchanged or computes the surprisal of events under its parent's model.

## Key concept: `p_node`

A `p_node` is a dict representing one P distribution in the chain. Structure:

```python
p_node = {
    "parent": None | p_node,       # None for the first node
    "mode": "passthru" | "surprisal",  # how events are transformed from parent
    "fenceposts": [],               # this node's interior bin edges (list of float)
    "events": np.array([]),         # the events this node received (after transform)
    "figure": bokeh Figure,         # the bar chart figure for this node
    "source": ColumnDataSource,     # data source for the bar chart
    "child": None | p_node,         # the next node in the chain, if any
    # UI widgets owned by this node:
    "derive_dropdown": Select,      # "Pass events thru as they are" / "Surprisal"
    "derive_btn": Button,           # "View derived distribution"
    "fencepost_input": TextInput,
    "fencepost_status": Div,
    "divide_bin_btn": Button,
    "equal_width_btn": Button,
    "ew_left_input": TextInput,
    "ew_right_input": TextInput,
    "ew_count_input": TextInput,
    "ew_submit_btn": Button,
    "ew_preview": Div,
    "ew_status": Div,
    "layout": Column,              # the Column containing this node's figure + controls
}
```

A module-level list tracks the chain:

```python
p_nodes = []  # ordered list; p_nodes[0] is the first (root) node
```

## Initial state

On load, the app shows only:
1. The event controls row: "Add events", n input, "Make distribution from events", "Clear events"
2. The rug plot
3. A "View derived distribution" button (no dropdown — the first node is always passthru)

There are no P distribution plots yet.

## Creating a new p_node

When the user clicks "View derived distribution" (on the last node in the chain, or the initial button if no nodes exist yet):

1. Create a new `p_node` dict.
2. Set `parent` to the previous node (or `None` if this is the first).
3. Set `mode`:
   - First node: always `"passthru"` (no dropdown is shown for it).
   - Subsequent nodes: read from the dropdown that was next to the button that was clicked. That dropdown is part of the *parent* node's UI... actually, see "UI per node" below for the cleaner version.
4. Copy `fenceposts` from parent (or `[]` if first node).
5. Create the bokeh figure, source, and all widget instances for this node.
6. Append the node's layout to the root Column.
7. Append to `p_nodes`.
8. If events exist, immediately run `recompute_chain()` so the new node shows a distribution.

## UI per node

Each p_node's layout Column contains, top to bottom:

1. **The P figure** — bar chart, shares x_range with rug plot. Title: `"P{i} | Entropy = X.XXXX bits"` where i is the 1-based index in the chain.
2. **Bin edge controls row** — "Add one bin edge" button + input + status, same as current.
3. **Equal width edges row** — "Add bin edges" button + inputs + submit + preview + status, same as current.
4. **Derive row** — contains:
   - A dropdown `Select` with options `["Pass events thru as they are", "Surprisal"]`.
     - For the **first** node: this dropdown is **not shown** (first node is always passthru).
     - For subsequent nodes: both options available, default to whatever was chosen at creation time. Changing it triggers `recompute_chain()`.
   - A "View derived distribution" button — clicking it creates the *next* node.

So the dropdown for node N's transform mode lives in node N's own layout (not its parent's). It just isn't shown for node 0.

## Event flow and `recompute_chain()`

This function is called whenever:
- "Make distribution from events" is clicked
- Events are added (after "Add events")
- A node's bin edges change (fencepost added, equal-width edges added)
- A node's mode dropdown changes

Algorithm:

```python
def recompute_chain():
    """Walk the chain, transforming and binning events at each step."""
    current_events = all_events  # raw events from the rug plot

    for node in p_nodes:
        # 1. Transform events based on mode
        if node["mode"] == "passthru" or node["parent"] is None:
            node["events"] = current_events
        else:  # surprisal
            parent = node["parent"]
            parent_edges = np.array([-np.inf] + sorted(parent["fenceposts"]) + [np.inf])
            parent_lefts, parent_rights, parent_probs = compute_probs(parent_edges, parent["events"])
            # For each event the parent received, compute -log2(P_parent(event))
            interior = parent_edges[1:-1]
            bin_indices = np.searchsorted(interior, parent["events"])
            surprisals = -np.log2(parent_probs[bin_indices])
            node["events"] = surprisals
            current_events = surprisals

        # 2. Bin events using this node's own edges
        edges = np.array([-np.inf] + sorted(node["fenceposts"]) + [np.inf])
        lefts, rights, probs = compute_probs(edges, node["events"])
        node["source"].data = make_source_data(
            lefts, rights, probs,
            x_start=rug_fig.x_range.start,
            x_end=rug_fig.x_range.end,
        )
        idx = p_nodes.index(node)
        node["figure"].title.text = f"P{idx+1}  |  Entropy = {entropy_bits(probs):.4f} bits"

        # 3. Set current_events for the next node in chain
        if node["mode"] == "passthru" or node["parent"] is None:
            current_events = node["events"]
        # (if surprisal, current_events was already set above)
```

**Important detail in the chain**: each node passes to its child either its received events (if child mode is passthru) or surprisal values computed from its own model + its received events (if child mode is surprisal). The transform is determined by the *child's* mode, applied using the *parent's* model. So the transform actually happens at the start of the child's step, looking back at the parent.

## Updating bin edges

When a bin edge is added to node N:
- Update `node["fenceposts"]`
- Call `recompute_chain()` (which recomputes all nodes — this is fine, it's cheap)

The bin edge change in node N affects node N's distribution and also affects any child whose mode is "surprisal" (since the parent's model changed). Recomputing the whole chain handles this correctly.

## "Make distribution from events" button

Currently this button triggers `refresh_p()`. Change it to call `recompute_chain()`. If `p_nodes` is empty, do nothing (the user needs to click "View derived distribution" first to create a node).

Actually, reconsider: it may be more intuitive if clicking "Make distribution from events" auto-creates the first p_node if none exist, then recomputes. Up to you — but at minimum it calls `recompute_chain()`.

## "Clear events" button

Clears `all_events`. Does NOT destroy p_nodes or their figures. Just clears the rug and sets each node's events to empty. Call `recompute_chain()` with empty events so all distributions reset to uniform.

## Infinite-edge JS callback

Each p_node's figure needs its own JS range callback (the `_range_cb` pattern) attached to the shared x_range, so that infinite-edge bars stretch on pan/zoom. Create this when creating the node's figure.

## Layout structure

```
root Column:
  top_controls Row (Add events, n=, Make distribution, Clear events)
  rug_fig
  "View derived distribution" button  <-- initial, before any nodes exist
  --- after first node created: ---
  p_nodes[0].layout  (figure + bin controls + derive row with btn)
  --- after second node created: ---
  p_nodes[1].layout  (figure + bin controls + dropdown + derive row with btn)
  ...
```

When a new node is created, the "View derived distribution" button from the previous node (or the initial one) should be removed/hidden since it's been "used". The new node gets its own "View derived distribution" button at the bottom of its layout.

Actually, simpler: keep the derive button as part of each node's layout. When clicked, it creates the child and disables itself (so you can't create two children from one parent). The initial "View derived distribution" button (before any nodes) is a standalone widget that gets removed from the root layout once the first node is created.

## Deletion / removing nodes

Out of scope for v1. Nodes are append-only.

## Summary of refactoring steps

1. **Extract `make_p_node()` factory function** — creates all widgets, figure, source, callbacks for one node. Returns the p_node dict.
2. **Extract `recompute_chain()`** — walks `p_nodes`, transforms events, updates sources and titles.
3. **Refactor callbacks**:
   - `cb_add_events` — unchanged, but also calls `recompute_chain()` after updating rug.
   - `cb_make_dist` — calls `recompute_chain()`.
   - `cb_clear_events` — clears events, calls `recompute_chain()`.
   - Bin edge callbacks become per-node (created inside `make_p_node`), each calls `recompute_chain()`.
   - New: `cb_derive(parent_node)` — creates a child node, appends to chain, updates layout.
   - New: `cb_mode_change(node)` — when dropdown changes, calls `recompute_chain()`.
4. **Remove global `fenceposts`** — each node has its own.
5. **Remove global `p_source`, `p_fig`** — each node has its own.
6. **Update layout** — root Column starts with just events section + initial derive button. Nodes are appended dynamically.
7. **Keep `compute_probs`, `make_source_data`, `entropy_bits`, `bar_colors`, `bin_edges` as-is** (make `bin_edges` take a fenceposts arg instead of using global).
