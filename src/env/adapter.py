from abc import ABC, abstractmethod


class EnvironmentAdapter(ABC):
    """Abstract bridge between raw environment and the Seam agent.

    Each environment (ALFWorld, ScienceWorld, and future benchmarks) implements this interface
    to translate its specific observation/action formats into the universal
    representations that Seam's causal graph and evolution engine expect.
    """

    # ─── Observation Parsing ───

    @abstractmethod
    def parse_observation(self, obs: str) -> dict:
        """Parse raw observation into structured state update.

        Returns dict with optional keys:
          arrived_at: str          — receptacle/location name agent moved to
          objects_here: list[str]  — object names visible at current location
          picked_up: bool         — whether agent picked something up
          put_down: bool          — whether agent placed something
          processed: str | None   — processing type done ("heat", "cool", "clean", None)
          used_device: bool       — whether a device was used/toggled
          error: str | None       — failure message if action failed
          target_visible: bool    — whether the target object is visible
        """

    # ─── Type Mapping ───

    @abstractmethod
    def obj_name_to_type(self, name: str) -> str:
        """Map raw object name to type string. E.g. 'egg' -> 'EggType'.
        Returns '' if unknown.
        """

    @abstractmethod
    def recep_name_to_type(self, name: str) -> str:
        """Map raw receptacle name to type string. E.g. 'counter' -> 'CounterType'.
        Returns '' if unknown.
        """

    # ─── Task Parsing ───

    @abstractmethod
    def parse_task(self, obs: str, task_type: str = "") -> dict:
        """Parse task description from initial observation.

        Returns:
          task_type: str        — canonical task type
          target_obj: str       — target object name
          target_obj_type: str  — target object type
          target_recep: str     — target receptacle name (if any)
          processing: str       — required processing ("heat", "cool", "clean", "use_lamp", "")
        """

    # ─── Action Matching ───

    @abstractmethod
    def action_matches_goto(self, action: str) -> str | None:
        """If action is a movement/goto, return destination name. Else None."""

    @abstractmethod
    def action_matches_take(self, action: str, target_obj: str) -> bool:
        """Check if action is taking the specified target object."""

    @abstractmethod
    def action_matches_process(self, action: str, process_type: str) -> bool:
        """Check if action performs the specified processing (heat/cool/clean/cook/etc)."""

    @abstractmethod
    def action_matches_put(self, action: str) -> bool:
        """Check if action is a put/place/drop action."""

    @abstractmethod
    def action_matches_use_device(self, action: str) -> bool:
        """Check if action uses/toggles a device."""

    def normalize_action(self, action: str, candidates: list[str] | None = None) -> str:
        """Normalize model-produced action text before validating against candidates.

        Environment-specific adapters can override this to convert common
        aliases into executable commands, e.g. TextWorld's "look at X" ->
        "examine X".
        """
        return action.strip()

    def allow_unlisted_action(self, action: str, candidates: list[str] | None = None) -> bool:
        """Whether an action can be sent even when absent from candidate actions."""
        return False

    # ─── Action Effect Extraction ───

    @abstractmethod
    def extract_action_effect(self, action: str) -> tuple[str, str] | None:
        """From a raw action string, extract (process_type, appliance_name).
        E.g. 'heat egg with microwave 1' -> ('heat', 'microwave')
        Returns None if not a processing action.
        """

    @abstractmethod
    def infer_appliance_type(self, appliance_name: str) -> str:
        """Map raw appliance name to type. E.g. 'microwave' -> 'MicrowaveType'."""

    # ─── Category System ───

    @abstractmethod
    def get_object_category(self, obj_type: str) -> str | None:
        """Map object type to semantic category. E.g. 'EggType' -> 'food'."""

    @abstractmethod
    def get_receptacle_category(self, recep_type: str) -> str | None:
        """Map receptacle type to semantic category. E.g. 'FridgeType' -> 'kitchen_storage'."""

    @abstractmethod
    def get_cold_start_appliance(self, process_type: str) -> str:
        """Cold-start fallback: which appliance name for a process type.
        E.g. 'heat' -> 'microwave'.
        """

    # ─── Prompt Configuration ───

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the base system prompt for this environment."""

    @abstractmethod
    def get_few_shot_key(self, task_type: str) -> str:
        """Map task_type to few-shot prompt key."""

    @abstractmethod
    def get_valid_commands(self) -> tuple[str, ...]:
        """Return tuple of valid command prefixes for action parsing."""

    @abstractmethod
    def get_action_list_text(self) -> str:
        """Return human-readable list of valid actions for the system prompt."""

    # ─── Reflection ───

    @abstractmethod
    def format_trajectory_for_reflection(self, trajectory: list[dict]) -> str:
        """Format trajectory entries into text for LLM reflection.
        Each entry has 'action' and 'observation' keys.
        """

    @abstractmethod
    def get_recep_from_action(self, action: str) -> str:
        """Extract receptacle/location name from a goto action. Returns '' if not goto."""
