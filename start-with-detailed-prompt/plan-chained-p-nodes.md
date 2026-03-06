# Plan: Chained P-Node Distributions

## Overview

Refactor `foo.py` so that the app starts with only the Events section (rug plot + event controls). The user can then create a chain of P distribution nodes, one at a time. Each node receives events, models them, and outputs either its received events unchanged or the surprisal of each event under its own model — depending on its `output_mode` setting.

## Key concept: `p_node`

A `p_node` is a dict representing one P distribution in the chain. Structure:

```python
p_node = {
    "output_mode": "passthru" | "surprisal",  # what this node passes to its child
    "interior_edges": [],               # this node's interior bin edges (list of float)
    "events": np.array([]),         # the events this node received (just events — it doesn't know their origin)
    "figure": bokeh Figure,         # the bar chart figure for this node
    "source": ColumnDataSource,     # data source for the bar chart
    "child": None | p_node,         # the next node in the chain, if any
    # UI widgets owned by this node:
    "derive_dropdown": Select,      # "Pass events thru as they are" / "Surprisal"
    "derive_btn": Button,           # "View derived distribution"
    "edge_input": TextInput,
    "edge_status": Div,
    "divide_bin_btn": Button,
    "equal_width_btn": Button,
    "equal_width_left_input": TextInput,
    "equal_width_right_input": TextInput,
    "equal_width_count_input": TextInput,
    "equal_width_submit_btn": Button,
    "equal_width_preview": Div,
    "equal_width_status": Div,
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
2. Set `output_mode` to `"passthru"` (default). The node's dropdown controls what it outputs to any future child.
3. Copy `interior_edges` from previous node (or `[]` if first node).
4. Set the previous node's `child` to this new node (if a previous node exists).
5. Create the bokeh figure, source, and all widget instances for this node.
6. Append the node's layout to the root Column.
7. Append to `p_nodes`.
8. If events exist, immediately run `recompute_from(new_node)` so the new node shows a distribution.

## UI per node

Each p_node's layout Column contains, top to bottom:

1. **The P figure** — bar chart, shares x_range with rug plot. Title: `"P{i} | Entropy = X.XXXX bits"` where i is the 1-based index in the chain.
2. **Bin edge controls row** — "Add one bin edge" button + input + status, same as current.
3. **Equal width edges row** — "Add bin edges" button + inputs + submit + preview + status, same as current.
4. **Derive row** — contains:
   - A dropdown `Select` with options `["Pass events thru as they are", "Surprisal"]`.
     - This controls what this node *outputs* to its child (i.e. sets `output_mode`).
     - Changing it triggers `recompute_from(child)` (if a child exists), since the child's input events have changed.
   - A "View derived distribution" button — clicking it creates the *next* node.

The dropdown for node N's output mode lives in node N's own layout. The child simply receives events — it doesn't know whether they are raw or surprisals.

## Event flow and `recompute_from(node)`

This is a recursive function. It recomputes the given node and then recurses into its child (if any). This means changes only cascade downward from the point of change.

It is called whenever:
- "Make distribution from events" is clicked — calls `recompute_from(p_nodes[0])`.
- A node's bin edges change — calls `recompute_from(that_node)`.
- A node's output mode dropdown changes — calls `recompute_from(that_node's child)` (since the child's input events changed).

It is **not** called when "Add events" is clicked. Adding events only updates the rug plot, same as current behavior. The user must click "Make distribution from events" to push events into the chain.

Algorithm:

```python
def recompute_from(node):
    """Recompute this node's distribution and push output to its child.

    node["events"] must already be set by the caller before calling this.
    The node does not look back — it only knows about its own events,
    edges, and output_mode. It is a singly-linked list (parent → child).
    """
    # 1. Bin events using this node's own edges
    edges = np.array([-np.inf] + sorted(node["interior_edges"]) + [np.inf])
    probs = compute_probabilities(edges, node["events"])
    node["source"].data = make_column_data_source_data(
        edges, probs,
        x_start=rug_fig.x_range.start,
        x_end=rug_fig.x_range.end,
    )
    idx = p_nodes.index(node)
    node["figure"].title.text = f"P{idx+1}  |  Entropy = {entropy_bits(probs):.4f} bits"

    # 2. Compute output events and push to child
    if node["child"] is not None:
        if node["output_mode"] == "passthru":
            node["child"]["events"] = node["events"]
        else:  # surprisal
            interior = edges[1:-1]
            bin_indices = np.searchsorted(interior, node["events"])
            node["child"]["events"] = -np.log2(probs[bin_indices])
        recompute_from(node["child"])
```

The root node's events are set directly by the caller: `p_nodes[0]["events"] = all_events` before calling `recompute_from(p_nodes[0])`.

**Important detail in the chain**: this is a singly-linked list. Each node only knows about its own events, edges, output_mode, and child. It never looks back at its parent. A node's `output_mode` controls what it computes and pushes forward to its child. The child simply receives events — it does not know or care whether they are raw events or surprisals.

## Updating bin edges

When a bin edge is added to node N:
- Update `node["interior_edges"]`
- Call `recompute_from(node)` — recomputes this node and cascades to all descendants.

The bin edge change in node N affects node N's distribution and also affects its output to any child (since the model changed, surprisal values change too). The recursive descent handles this correctly.

## "Make distribution from events" button

Currently this button triggers `refresh_p()`. Change it to call `recompute_from(p_nodes[0])`. If `p_nodes` is empty, do nothing (the user needs to click "View derived distribution" first to create a node).

Actually, reconsider: it may be more intuitive if clicking "Make distribution from events" auto-creates the first p_node if none exist, then recomputes. Up to you — but at minimum it calls `recompute_from(p_nodes[0])`.

## "Clear events" button

Clears `all_events`. Does NOT destroy p_nodes or their figures. Just clears the rug and sets each node's events to empty. Call `recompute_from(p_nodes[0])` (if any nodes exist) so all distributions reset to uniform.

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
2. **Extract `recompute_from(node)`** — recursive function that recomputes the given node and all descendants.
3. **Refactor callbacks**:
   - `on_add_events` — unchanged. Does NOT trigger recomputation (same as current behavior).
   - `on_make_dist` — calls `recompute_from(p_nodes[0])`.
   - `on_clear_events` — clears events, calls `recompute_from(p_nodes[0])`.
   - Bin edge callbacks become per-node (created inside `make_p_node`), each calls `recompute_from(that_node)`.
   - New: `on_derive(parent_node)` — creates a child node, appends to chain, updates layout.
   - New: `on_output_mode_change(node)` — when dropdown changes, updates `node["output_mode"]` and calls `recompute_from(node["child"])` if a child exists.
4. **Remove global `interior_edges`** — each node has its own.
5. **Remove global `p_column_data_source`, `p_fig`** — each node has its own.
6. **Update layout** — root Column starts with just events section + initial derive button. Nodes are appended dynamically.
7. **Keep `compute_probabilities`, `make_column_data_source_data`, `entropy_bits`, `bar_colors`, `bin_edges` as-is** (make `bin_edges` take an interior_edges arg instead of using global).
