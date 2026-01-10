"""Tests for the deployment agent and targets."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple test model for deployment tests."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def simple_onnx_model(tmp_path):
    """Create a simple ONNX model for testing."""
    model = SimpleModel()
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)

    onnx_path = tmp_path / "test_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    return onnx_path


@pytest.fixture
def simple_pytorch_model(tmp_path):
    """Create a simple PyTorch model file for testing."""
    model = SimpleModel()
    model_path = tmp_path / "test_model.pt"
    torch.save(model, model_path)
    return model_path


class TestDeploymentModels:
    """Tests for deployment data models."""

    def test_deployment_precision_enum(self):
        """Test DeploymentPrecision enum values."""
        from embodied_ai_architect.agents.deployment.models import DeploymentPrecision

        assert DeploymentPrecision.FP32.value == "fp32"
        assert DeploymentPrecision.FP16.value == "fp16"
        assert DeploymentPrecision.INT8.value == "int8"

    def test_calibration_config(self):
        """Test CalibrationConfig model."""
        from embodied_ai_architect.agents.deployment.models import CalibrationConfig

        config = CalibrationConfig(
            data_path=Path("/data/calib"),
            num_samples=50,
            batch_size=4,
            preprocessing="yolo",
        )
        assert config.num_samples == 50
        assert config.batch_size == 4
        assert config.preprocessing == "yolo"

    def test_validation_config_defaults(self):
        """Test ValidationConfig default values."""
        from embodied_ai_architect.agents.deployment.models import ValidationConfig

        config = ValidationConfig()
        assert config.num_samples == 100
        assert config.tolerance_percent == 1.0
        assert config.compare_outputs is True

    def test_deployment_artifact(self):
        """Test DeploymentArtifact model."""
        from embodied_ai_architect.agents.deployment.models import (
            DeploymentArtifact,
            DeploymentPrecision,
        )

        artifact = DeploymentArtifact(
            engine_path=Path("/output/model.engine"),
            precision=DeploymentPrecision.INT8,
            target="jetson",
            input_shape=(1, 3, 224, 224),
            output_shape=(1, 1000),
            size_bytes=1024000,
        )
        assert artifact.precision == DeploymentPrecision.INT8
        assert artifact.size_bytes == 1024000


class TestDeploymentAgent:
    """Tests for the DeploymentAgent."""

    def test_agent_init_no_targets(self):
        """Test agent initializes without available targets."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        # Mock the auto-discovery to return no targets
        with patch.object(DeploymentAgent, "_auto_discover_targets"):
            agent = DeploymentAgent()
            # Should have empty targets dict since we mocked discovery
            assert isinstance(agent.targets, dict)

    def test_agent_list_targets(self):
        """Test listing available targets."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()
        targets = agent.list_targets()
        assert isinstance(targets, list)

    def test_agent_requires_model(self):
        """Test agent fails gracefully without model."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()
        result = agent.execute({})
        assert not result.success
        assert "No model provided" in result.error

    def test_agent_requires_input_shape(self, simple_onnx_model):
        """Test agent fails gracefully without input shape."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()
        result = agent.execute({"model": str(simple_onnx_model)})
        assert not result.success
        assert "input_shape" in result.error.lower()

    def test_agent_requires_available_target(self, simple_onnx_model):
        """Test agent fails gracefully with unavailable target."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        # Mock to have no targets
        with patch.object(DeploymentAgent, "_auto_discover_targets"):
            agent = DeploymentAgent()
            result = agent.execute({
                "model": str(simple_onnx_model),
                "target": "nonexistent",
                "input_shape": [1, 3, 32, 32],
            })
            assert not result.success
            assert "not available" in result.error.lower()


class TestDeploymentTargetBase:
    """Tests for the DeploymentTarget base class."""

    def test_base_target_abstract(self):
        """Test that DeploymentTarget is abstract."""
        from embodied_ai_architect.agents.deployment.targets.base import DeploymentTarget

        with pytest.raises(TypeError):
            DeploymentTarget("test")


# Only run OpenVINO tests if available
try:
    import openvino  # noqa: F401
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False


@pytest.mark.skipif(not HAS_OPENVINO, reason="OpenVINO not installed")
class TestOpenVINOTarget:
    """Tests for the OpenVINO deployment target."""

    def test_openvino_is_available(self):
        """Test OpenVINO availability detection."""
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget

        target = OpenVINOTarget()
        assert target.is_available() is True

    def test_openvino_capabilities(self):
        """Test OpenVINO capabilities reporting."""
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget

        target = OpenVINOTarget()
        caps = target.get_capabilities()

        assert caps["name"] == "openvino"
        assert "fp32" in caps["supported_precisions"]
        assert "fp16" in caps["supported_precisions"]
        assert "int8" in caps["supported_precisions"]
        assert caps["supports_calibration"] is True
        assert "openvino_version" in caps

    def test_openvino_deploy_fp32(self, simple_onnx_model, tmp_path):
        """Test FP32 deployment with OpenVINO."""
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget
        from embodied_ai_architect.agents.deployment.models import DeploymentPrecision

        target = OpenVINOTarget()
        output_path = tmp_path / "model_fp32.xml"

        artifact = target.deploy(
            model=simple_onnx_model,
            precision=DeploymentPrecision.FP32,
            output_path=output_path,
            input_shape=(1, 3, 32, 32),
        )

        assert artifact.engine_path.exists()
        assert artifact.precision == DeploymentPrecision.FP32
        assert artifact.target == "openvino"
        assert artifact.size_bytes > 0

    def test_openvino_deploy_fp16(self, simple_onnx_model, tmp_path):
        """Test FP16 deployment with OpenVINO."""
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget
        from embodied_ai_architect.agents.deployment.models import DeploymentPrecision

        target = OpenVINOTarget()
        output_path = tmp_path / "model_fp16.xml"

        artifact = target.deploy(
            model=simple_onnx_model,
            precision=DeploymentPrecision.FP16,
            output_path=output_path,
            input_shape=(1, 3, 32, 32),
        )

        assert artifact.engine_path.exists()
        assert artifact.precision == DeploymentPrecision.FP16

    def test_openvino_validate(self, simple_onnx_model, tmp_path):
        """Test validation with OpenVINO."""
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget
        from embodied_ai_architect.agents.deployment.models import (
            DeploymentPrecision,
            ValidationConfig,
        )

        target = OpenVINOTarget()

        # Deploy first
        artifact = target.deploy(
            model=simple_onnx_model,
            precision=DeploymentPrecision.FP32,
            output_path=tmp_path / "model.xml",
            input_shape=(1, 3, 32, 32),
        )

        # Validate
        val_config = ValidationConfig(num_samples=5, tolerance_percent=1.0)
        result = target.validate(artifact, simple_onnx_model, val_config)

        assert result.passed is True
        assert result.samples_compared == 5
        assert result.baseline_latency_ms is not None
        assert result.deployed_latency_ms is not None
        assert result.max_output_diff is not None
        assert result.max_output_diff < 0.01  # FP32 should be very close

    def test_openvino_int8_requires_calibration(self, simple_onnx_model, tmp_path):
        """Test INT8 deployment requires calibration config."""
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget
        from embodied_ai_architect.agents.deployment.models import DeploymentPrecision

        target = OpenVINOTarget()

        with pytest.raises(ValueError, match="calibration"):
            target.deploy(
                model=simple_onnx_model,
                precision=DeploymentPrecision.INT8,
                output_path=tmp_path / "model.xml",
                input_shape=(1, 3, 32, 32),
                calibration=None,
            )


@pytest.mark.skipif(not HAS_OPENVINO, reason="OpenVINO not installed")
class TestDeploymentAgentWithOpenVINO:
    """Integration tests for DeploymentAgent with OpenVINO."""

    def test_full_deployment_workflow(self, simple_onnx_model, tmp_path):
        """Test complete deployment workflow through the agent."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()

        # Skip if no targets available
        if not agent.list_targets():
            pytest.skip("No deployment targets available")

        result = agent.execute({
            "model": str(simple_onnx_model),
            "target": "openvino",
            "precision": "fp32",
            "input_shape": [1, 3, 32, 32],
            "output_dir": str(tmp_path),
        })

        assert result.success, f"Deployment failed: {result.error}"
        assert "artifact" in result.data
        assert result.data["artifact"]["precision"] == "fp32"

    def test_pytorch_to_onnx_export(self, simple_pytorch_model, tmp_path):
        """Test automatic PyTorch to ONNX export."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()

        if not agent.list_targets():
            pytest.skip("No deployment targets available")

        result = agent.execute({
            "model": str(simple_pytorch_model),
            "target": "openvino",
            "precision": "fp32",
            "input_shape": [1, 3, 32, 32],
            "output_dir": str(tmp_path),
        })

        assert result.success, f"Deployment failed: {result.error}"
        # Should have exported to ONNX first
        assert "Exporting to ONNX" in str(result.data.get("logs", []))
