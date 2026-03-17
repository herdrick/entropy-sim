# Plan: Chained P-Node Distributions

## Overview

Refactor `foo.py` so that the app starts with only the Events section (rug plot + event controls). The user can then create a chain of P distribution nodes, one at a time. Each node receives events, models them, and outputs either its received events unchanged or the surprisal of each event under its own model — depending on its `output_mode` setting.

## Key concept: `PNode`

A `PNode` is a dataclass representing one P distribution in the chain. The chain is a singly-linked list — there is no separate list of nodes.

```python
@dataclass
class PNode:
    output_mode: str = "passthru"       # "passthru" | "surprisal" — what this node passes to its child
    interior_edges: list = field(default_factory=list)  # interior bin edges (sorted)
    events: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))  # received events
    figure: figure = None               # the bar chart figure for this node
    source: ColumnDataSource = None     # data source for the bar chart
    rug_fig: figure = None              # rug plot showing this node's received events
    rug_source: ColumnDataSource = None # data source for the rug plot
    child: Optional["PNode"] = None     # the next node in the chain, if any
    # UI widgets owned by this node:
    derive_dropdown: Select = None      # "Pass events thru as they are" / "Surprisal"
    derive_btn: Button = None           # "View derived distribution"
    edge_input: TextInput = None
    edge_status: Div = None
    divide_bin_btn: Button = None
    equal_width_btn: Button = None
    equal_width_left_input: TextInput = None
    equal_width_right_input: TextInput = None
    equal_width_count_input: TextInput = None
    equal_width_submit_btn: Button = None
    equal_width_preview: Div = None
    equal_width_status: Div = None
    layout: Column = None               # the Column containing this node's figure + controls
```

A module-level variable points to the head of the chain:

```python
root_node: Optional[PNode] = None  # head of the singly-linked list; None before first node is created
```

## Initial state

On load, the app shows only:
1. The event controls row: "Add events", n input, "Make distribution from events", "Clear events"
2. The rug plot
3. A "View derived distribution" button (no dropdown — the first node is always passthru)

There are no P distribution plots yet.

## Creating a new PNode

When the user clicks "View derived distribution" (on the last node in the chain, or the initial button if no nodes exist yet):

1. Create a new `PNode` (interior_edges starts empty — one big bin).
2. Set `output_mode` to `"passthru"` (default). The node's dropdown controls what it outputs to any future child.
3. Set the previous node's `child` to this new node (if a previous node exists). If no previous node, set `root_node` to this node.
4. Create the bokeh figure (with its own independent x_range), source, and all widget instances for this node.
5. Append the node's layout to the root Column.
6. If events exist, immediately run `recompute_from(new_node)` so the new node shows a distribution.

## UI per node

Each PNode's layout Column contains, top to bottom:

1. **Rug plot** — each P node has its own rug plot showing the events it received, with title `"Events (<n>)"` where n is the count. This lets the user see the node's intake of events. Shares x_range with the node's bar chart figure.
2. **The P figure** — bar chart with its own independent x_range (since surprisal nodes operate on a completely different domain than the raw events). Title: `"P{i} | Entropy = X.XXXX bits"` where i is the 1-based index in the chain.
3. **Bin edge controls row** — "Add one bin edge" button + input + status, same as current.
4. **Equal width edges row** — "Add bin edges" button + inputs + submit + preview + status, same as current.
5. **Derive row** — contains:
   - A dropdown `Select` with options `["Pass events thru as they are", "Surprisal"]`.
     - This controls what this node *outputs* to its child (i.e. sets `output_mode`).
     - Changing it triggers `recompute_from(child)` (if a child exists), since the child's input events have changed.
   - A "View derived distribution" button — clicking it creates the *next* node.

The dropdown for node N's output mode lives in node N's own layout. The child simply receives events — it doesn't know whether they are raw or surprisals.

## Event flow and `recompute_from(node)`

This is a recursive function. It recomputes the given node and then recurses into its child (if any). This means changes only cascade downward from the point of change.

It is called whenever:
- "Make distribution from events" is clicked — calls `recompute_from(root_node)`.
- A node's bin edges change — calls `recompute_from(that_node)`.
- A node's output mode dropdown changes — calls `recompute_from(that_node.child)` (since the child's input events changed).

It is **not** called when "Add events" is clicked. Adding events only updates the rug plot, same as current behavior. The user must click "Make distribution from events" to push events into the chain.

Algorithm:

```python
def node_index(node):
    """Walk the chain from root_node to find this node's 0-based index."""
    idx, cur = 0, root_node
    while cur is not node:
        cur = cur.child
        idx += 1
    return idx

def recompute_from(node):
    """Recompute this node's distribution and push output to its child.

    node.events must already be set by the caller before calling this.
    The node does not look back — it only knows about its own events,
    edges, and output_mode. It is a singly-linked list (parent → child).
    """
    # 1. Bin events using this node's own edges
    edges = np.array([-np.inf] + sorted(node.interior_edges) + [np.inf])
    probs = compute_probabilities(edges, node.events)
    node.source.data = make_column_data_source_data(
        edges, probs,
        x_start=node.figure.x_range.start,
        x_end=node.figure.x_range.end,
    )
    idx = node_index(node)
    node.figure.title.text = f"P{idx+1}  |  Entropy = {entropy_bits(probs):.4f} bits"

    # 2. Compute output events and push to child
    if node.child is not None:
        if node.output_mode == "passthru":
            node.child.events = node.events
        else:  # surprisal
            interior = edges[1:-1]
            bin_indices = np.searchsorted(interior, node.events)
            node.child.events = -np.log2(probs[bin_indices])
        recompute_from(node.child)
```

The root node's events are set directly by the caller: `root_node.events = root_events` before calling `recompute_from(root_node)`.

**Important detail in the chain**: this is a singly-linked list. Each node only knows about its own events, edges, output_mode, and child. It never looks back at its parent. A node's `output_mode` controls what it computes and pushes forward to its child. The child simply receives events — it does not know or care whether they are raw events or surprisals.

## Updating bin edges

When a bin edge is added to node N:
- Update `node.interior_edges`
- Call `recompute_from(node)` — recomputes this node and cascades to all descendants.

The bin edge change in node N affects node N's distribution and also affects its output to any child (since the model changed, surprisal values change too). The recursive descent handles this correctly.

## "Make distribution from events" button

Currently this button triggers `refresh_p()`. Change it to call `recompute_from(root_node)`. If `root_node` is None, "Make distribution from events" should be disabled.

## "Clear events" button

Clears `root_events`. Does NOT destroy PNodes or their figures. Just clears the rug plot.

## Infinite-edge JS callback

Each PNode's figure needs its own JS range callback (the `_range_cb` pattern) attached to its own x_range, so that infinite-edge bars stretch on pan/zoom. Create this when creating the node's figure.

## Layout structure

```
root Column:
  top_controls Row (Add events, n=, Make distribution, Clear events)
  rug_fig
  "View derived distribution" button  <-- initial, before any nodes exist
  --- after first node created: ---
  root_node.layout  (figure + bin controls + derive row with btn)
  --- after second node created: ---
  root_node.child.layout  (figure + bin controls + dropdown + derive row with btn)
  ...
```

When clicked, the "View derived distribution" button creates a distribution and then disables itself (so you can't create two children from one parent).

## Deletion / removing nodes

Out of scope for v1. Nodes are append-only.

## Summary of refactoring steps

1. **Define `PNode` dataclass** — holds all per-node state, widgets, figure, source, and `child` pointer.
2. **Extract `make_p_node()` factory function** — creates a `PNode` with all widgets, figure (with its own x_range), source, and callbacks. Returns the `PNode`.
3. **Extract `recompute_from(node)`** — recursive function that recomputes the given node and all descendants.
4. **Refactor callbacks**:
   - `on_add_events` — unchanged. Does NOT trigger recomputation (same as current behavior).
   - `on_make_dist` — calls `recompute_from(root_node)`.
   - `on_clear_events` — clears events, clears rug plot.
   - Bin edge callbacks become per-node (created inside `make_p_node`), each calls `recompute_from(that_node)`.
   - New: `on_derive(parent_node)` — creates a child node, links it, updates layout.
   - New: `on_output_mode_change(node)` — when dropdown changes, updates `node.output_mode` and calls `recompute_from(node.child)` if a child exists.
5. **Remove global `interior_edges`** — each node has its own.
6. **Remove global `p_column_data_source`, `p_fig`** — each node has its own.
7. **Remove `p_nodes` list** — use `root_node` and walk the linked list via `.child`.
8. **Update layout** — root Column starts with just events section + initial derive button. Nodes are appended dynamically.
9. **Keep `compute_probabilities`, `make_column_data_source_data`, `entropy_bits`, `bar_colors`, `bin_edges` as-is** (make `bin_edges` take an interior_edges arg instead of using global).
