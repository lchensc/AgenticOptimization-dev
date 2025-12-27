"""
Pydantic schemas for evaluator registration configurations.

These schemas define the structure of evaluator configurations stored in Foundry.
No code generation - just metadata that FoundryEvaluator uses to execute evaluations.
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class EvaluatorSource(BaseModel):
    """
    Source configuration - where the evaluator comes from.

    Supports:
    - python_function: Python callable in a file
    - cli_executable: Command-line tool
    - api_endpoint: HTTP API (future)
    - workflow: Multi-step workflow (future)
    """

    type: Literal["python_function", "cli_executable", "api_endpoint", "workflow"] = Field(
        ...,
        description="Type of evaluator source"
    )

    # Python function fields
    file_path: Optional[str] = Field(
        None,
        description="Path to Python file containing evaluator"
    )
    callable_name: Optional[str] = Field(
        None,
        description="Name of callable function/class"
    )
    gradient_callable: Optional[str] = Field(
        None,
        description="Name of gradient function (optional)"
    )

    # CLI executable fields
    command: Optional[str] = Field(
        None,
        description="Command to execute"
    )
    input_file: Optional[str] = Field(
        None,
        description="Input file path for design variables"
    )
    output_file: Optional[str] = Field(
        None,
        description="Output file path for results"
    )
    input_format: Optional[Literal["text", "json", "csv", "binary"]] = Field(
        None,
        description="Format for input data"
    )
    output_format: Optional[Literal["text", "json", "csv", "stdout", "binary"]] = Field(
        None,
        description="Format for output data"
    )

    # Common fields
    working_directory: Optional[str] = Field(
        None,
        description="Working directory for execution"
    )

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v, info):
        """Validate file_path for python_function."""
        if info.data.get('type') == 'python_function' and not v:
            raise ValueError("file_path required for python_function type")
        return v

    @field_validator('callable_name')
    @classmethod
    def validate_callable_name(cls, v, info):
        """Validate callable_name for python_function."""
        if info.data.get('type') == 'python_function' and not v:
            raise ValueError("callable_name required for python_function type")
        return v

    @field_validator('command')
    @classmethod
    def validate_command(cls, v, info):
        """Validate command for cli_executable."""
        if info.data.get('type') == 'cli_executable' and not v:
            raise ValueError("command required for cli_executable type")
        return v

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


class InputInterface(BaseModel):
    """Input interface specification."""

    type: Literal["numpy_array", "dict", "dataframe"] = Field(
        default="numpy_array",
        description="Input data type"
    )
    expected_shape: Optional[tuple] = Field(
        None,
        description="Expected input shape (if known)"
    )
    validated_shapes: List[tuple] = Field(
        default_factory=list,
        description="Shapes validated during testing"
    )

    class Config:
        extra = "allow"


class OutputInterface(BaseModel):
    """Output interface specification."""

    format: Literal["scalar", "dict", "tuple", "array", "auto"] = Field(
        default="auto",
        description="Return value format"
    )
    n_outputs: int = Field(
        default=1,
        description="Number of output values (explicit declaration, no guessing)"
    )
    output_names: Optional[List[str]] = Field(
        None,
        description="Names for each output (e.g., ['f1', 'f2'] or ['drag', 'lift'])"
    )
    keys: Optional[List[str]] = Field(
        None,
        description="Dictionary keys (if format is dict)"
    )
    types: Optional[Dict[str, str]] = Field(
        None,
        description="Types of return values"
    )

    class Config:
        extra = "allow"


class GradientInterface(BaseModel):
    """Gradient computation specification."""

    available: bool = Field(
        default=False,
        description="Whether user provides gradients"
    )
    method: Literal["user_provided", "finite_difference", "adjoint", "auto"] = Field(
        default="finite_difference",
        description="Gradient computation method"
    )

    class Config:
        extra = "allow"


class EvaluatorInterface(BaseModel):
    """
    Discovered interface specification.

    LLM discovers this by testing the evaluator with dummy inputs.
    """

    input: InputInterface = Field(
        default_factory=InputInterface,
        description="Input interface"
    )
    output: OutputInterface = Field(
        default_factory=OutputInterface,
        description="Output interface"
    )
    gradients: GradientInterface = Field(
        default_factory=GradientInterface,
        description="Gradient interface"
    )

    class Config:
        extra = "allow"


class EvaluatorCapabilities(BaseModel):
    """
    PAOLA capabilities configuration.

    These are built into FoundryEvaluator infrastructure.
    """

    observation_gates: bool = Field(
        default=True,
        description="Enable observation before/after evaluations"
    )
    interjection_enabled: bool = Field(
        default=True,
        description="Allow user/agent to interrupt evaluations"
    )
    caching: bool = Field(
        default=True,
        description="Enable evaluation caching"
    )
    cost_tracking: bool = Field(
        default=True,
        description="Track computational cost"
    )
    parallel_safe: bool = Field(
        default=False,
        description="Safe to run in parallel"
    )
    deterministic: bool = Field(
        default=True,
        description="Same input always gives same output"
    )

    class Config:
        extra = "allow"


class EvaluatorPerformance(BaseModel):
    """
    Performance metrics (learned over time).

    Updated after each evaluation.
    """

    median_time: Optional[float] = Field(
        None,
        description="Median execution time (seconds)"
    )
    std_time: Optional[float] = Field(
        None,
        description="Standard deviation of execution time"
    )
    cost_per_eval: float = Field(
        default=1.0,
        description="Computational cost per evaluation"
    )
    success_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Success rate (0-1)"
    )
    total_calls: int = Field(
        default=0,
        ge=0,
        description="Total number of evaluations"
    )

    class Config:
        extra = "allow"


class EvaluatorLineage(BaseModel):
    """
    Lineage tracking - where this evaluator came from and where it's used.
    """

    registered_at: Optional[datetime] = Field(
        None,
        description="When evaluator was registered"
    )
    registered_by: Optional[str] = Field(
        None,
        description="Who registered (user, agent, system)"
    )
    used_in_runs: List[str] = Field(
        default_factory=list,
        description="Run IDs that use this evaluator"
    )
    used_in_problems: List[str] = Field(
        default_factory=list,
        description="Problem IDs that use this evaluator"
    )
    parent_evaluator_id: Optional[str] = Field(
        None,
        description="Parent evaluator if this is derived"
    )

    class Config:
        extra = "allow"


class EvaluatorMetadata(BaseModel):
    """Additional metadata about the evaluator."""

    description: Optional[str] = Field(
        None,
        description="Human-readable description"
    )
    domain: Optional[str] = Field(
        None,
        description="Problem domain (CFD, ML, finance, etc.)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for searching/filtering"
    )
    version: Optional[str] = Field(
        None,
        description="Evaluator version"
    )

    class Config:
        extra = "allow"


class EvaluatorConfig(BaseModel):
    """
    Complete evaluator configuration.

    This is what gets stored in Foundry - NOT Python wrapper code!
    Just metadata that FoundryEvaluator uses for execution.
    """

    evaluator_id: str = Field(
        ...,
        description="Unique identifier for this evaluator"
    )
    name: str = Field(
        ...,
        description="Human-readable name"
    )
    status: Literal["registered", "validated", "active", "failed", "deprecated"] = Field(
        default="registered",
        description="Current status"
    )

    # Core configuration
    source: EvaluatorSource = Field(
        ...,
        description="Where the evaluator comes from"
    )
    interface: EvaluatorInterface = Field(
        default_factory=EvaluatorInterface,
        description="Discovered interface specification"
    )
    capabilities: EvaluatorCapabilities = Field(
        default_factory=EvaluatorCapabilities,
        description="PAOLA capabilities configuration"
    )

    # Runtime data
    performance: EvaluatorPerformance = Field(
        default_factory=EvaluatorPerformance,
        description="Performance metrics"
    )
    lineage: EvaluatorLineage = Field(
        default_factory=EvaluatorLineage,
        description="Lineage tracking"
    )
    metadata: EvaluatorMetadata = Field(
        default_factory=EvaluatorMetadata,
        description="Additional metadata"
    )

    class Config:
        extra = "allow"  # Allow additional fields
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

    def dict(self, **kwargs):
        """Override to handle datetime serialization."""
        d = super().dict(**kwargs)
        # Ensure lineage.registered_at is serialized
        if 'lineage' in d and 'registered_at' in d['lineage']:
            if isinstance(d['lineage']['registered_at'], datetime):
                d['lineage']['registered_at'] = d['lineage']['registered_at'].isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluatorConfig":
        """
        Create from dictionary (e.g., loaded from JSON).

        Handles datetime parsing.
        """
        # Parse registered_at if it's a string
        if 'lineage' in data and 'registered_at' in data['lineage']:
            if isinstance(data['lineage']['registered_at'], str):
                data['lineage']['registered_at'] = datetime.fromisoformat(
                    data['lineage']['registered_at']
                )

        return cls(**data)


# Convenience functions for creating configurations

def create_python_function_config(
    evaluator_id: str,
    name: str,
    file_path: str,
    callable_name: str,
    gradient_callable: Optional[str] = None,
    **kwargs
) -> EvaluatorConfig:
    """
    Create configuration for Python function evaluator.

    Args:
        evaluator_id: Unique ID
        name: Human-readable name
        file_path: Path to Python file
        callable_name: Function name
        gradient_callable: Optional gradient function name
        **kwargs: Additional configuration

    Returns:
        EvaluatorConfig instance
    """
    source = EvaluatorSource(
        type="python_function",
        file_path=file_path,
        callable_name=callable_name,
        gradient_callable=gradient_callable
    )

    return EvaluatorConfig(
        evaluator_id=evaluator_id,
        name=name,
        source=source,
        **kwargs
    )


def create_cli_executable_config(
    evaluator_id: str,
    name: str,
    command: str,
    input_file: str,
    output_file: str,
    input_format: str = "text",
    output_format: str = "text",
    **kwargs
) -> EvaluatorConfig:
    """
    Create configuration for CLI executable evaluator.

    Args:
        evaluator_id: Unique ID
        name: Human-readable name
        command: Command to execute
        input_file: Input file path
        output_file: Output file path
        input_format: Input data format
        output_format: Output data format
        **kwargs: Additional configuration

    Returns:
        EvaluatorConfig instance
    """
    source = EvaluatorSource(
        type="cli_executable",
        command=command,
        input_file=input_file,
        output_file=output_file,
        input_format=input_format,
        output_format=output_format
    )

    return EvaluatorConfig(
        evaluator_id=evaluator_id,
        name=name,
        source=source,
        **kwargs
    )
