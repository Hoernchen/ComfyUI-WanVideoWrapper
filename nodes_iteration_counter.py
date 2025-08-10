"""
Loop detection only checks static graph, not dynamic execution.
"""

import logging
from typing import Tuple, Dict, Any, Optional

log = logging.getLogger("WanIterationCounter")


class AnyType(str):
    """A special type that accepts any input type by always returning True for equality checks."""
    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class WanIterationCounter:
    """
    Simple iteration counter that breaks circular dependencies.

    This node examines WanVideoLoopedGeneration toggle states without
    creating execution dependencies, following the Easy-Use pattern.
    """

    @classmethod
    def INPUT_TYPES(cls):
        any_type = AnyType("*")
        return {
            "required": {
                # We only need the looped node connection to read its state
            },
            "optional": {
                "looped_node": (any_type, {
                    "tooltip": "Connect from WanVideoLoopedGeneration to count enabled iterations"
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("total_iterations",)
    FUNCTION = "count_iterations"
    CATEGORY = "WanVideoWrapper/Loop"
    DESCRIPTION = "Counts enabled prompt pairs from WanVideoLoopedGeneration without circular dependencies"

    # Critical: This allows execution without dependencies
    OUTPUT_NODE = True

    def count_iterations(self, prompt=None, unique_id=None, looped_node=None, **kwargs):
        """
        Count enabled iterations by examining the prompt structure.

        This works because ComfyUI passes the entire prompt structure,
        including widget values for all nodes in the graph.
        """

        if not prompt:
            log.warning("No prompt data available, defaulting to 1 iteration")
            return (1,)

        # Find the WanVideoLoopedGeneration node
        looped_node_data = self._find_looped_node(prompt, unique_id)

        if not looped_node_data:
            log.warning("No WanVideoLoopedGeneration node found, defaulting to 1 iteration")
            return (1,)

        # Count enabled prompt pairs
        enabled_count = self._count_enabled_pairs(looped_node_data)

        # Ensure at least 1 iteration
        total_iterations = max(1, enabled_count)

        log.info(f"WanIterationCounter: Found {total_iterations} enabled iterations")
        return (total_iterations,)

    def _find_looped_node(self, prompt: Dict, unique_id: Optional[str]) -> Optional[Dict]:
        """Find the WanVideoLoopedGeneration node in the prompt."""

        # First try to find by connection if we have a unique_id
        if unique_id and unique_id in prompt:
            node_data = prompt[unique_id]
            if "inputs" in node_data:
                for input_value in node_data["inputs"].values():
                    if isinstance(input_value, list) and len(input_value) == 2:
                        # This is a connection [node_id, slot]
                        connected_node_id = str(input_value[0])
                        if connected_node_id in prompt:
                            connected_node = prompt[connected_node_id]
                            if connected_node.get("class_type") == "WanVideoLoopedGeneration":
                                return connected_node

        # If not found by connection, search all nodes
        for node in prompt.values():
            if node.get("class_type") == "WanVideoLoopedGeneration":
                return node

        return None

    def _count_enabled_pairs(self, looped_node_data: Dict) -> int:
        """Count enabled prompt pairs in the looped node."""

        inputs = looped_node_data.get("inputs", {})
        enabled_count = 0

        # Check up to MAX_PROMPTS (20) prompt pairs
        for i in range(1, 21):
            pos_key = f"positive_{i:02d}"
            toggle_key = f"toggle_{i}"

            # Check if positive prompt exists and has a connection
            pos_value = inputs.get(pos_key)
            if pos_value is None:
                continue

            # If it's a list, it's a connection [node_id, slot]
            has_positive = isinstance(pos_value, list) and len(pos_value) == 2

            if has_positive:
                # Check toggle state (default to enabled)
                enabled = True
                toggle_value = inputs.get(toggle_key)

                if toggle_value is not None:
                    if isinstance(toggle_value, dict):
                        enabled = toggle_value.get('enabled', True)
                    elif isinstance(toggle_value, bool):
                        enabled = toggle_value

                if enabled:
                    enabled_count += 1
                    log.debug(f"Pair {i}: Enabled")
                else:
                    log.debug(f"Pair {i}: Disabled by toggle")

        return enabled_count

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-evaluate to catch toggle state changes."""
        return float("NaN")


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanIterationCounter": WanIterationCounter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanIterationCounter": "WAN Iteration Counter",
}