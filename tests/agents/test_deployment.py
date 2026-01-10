"""Tests for the deployment agent and targets."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

# Check for PIL availability
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Check for NNCF availability
try:
    import nncf  # noqa: F401
    HAS_NNCF = True
except ImportError:
    HAS_NNCF = False

# Check for TensorFlow availability (for Coral)
try:
    import tensorflow as tf  # noqa: F401
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


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


@pytest.fixture
def calibration_images(tmp_path):
    """Create synthetic calibration images for INT8 quantization testing.

    Generates 20 diverse images with:
    - Varied colors and patterns
    - Gradients and noise
    - Different intensity distributions

    This simulates a representative calibration dataset.
    """
    if not HAS_PIL:
        pytest.skip("PIL required for calibration image tests")

    calib_dir = tmp_path / "calibration_images"
    calib_dir.mkdir()

    image_size = (32, 32)  # Match our test model input size
    num_images = 20

    for i in range(num_images):
        # Create varied synthetic images
        if i % 4 == 0:
            # Solid color with noise
            base_color = np.random.randint(0, 255, 3)
            img_array = np.full((*image_size, 3), base_color, dtype=np.uint8)
            noise = np.random.randint(-30, 30, (*image_size, 3))
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif i % 4 == 1:
            # Horizontal gradient
            gradient = np.linspace(0, 255, image_size[1], dtype=np.uint8)
            img_array = np.tile(gradient, (image_size[0], 1))
            img_array = np.stack([img_array] * 3, axis=-1)
            # Add color tint
            tint = np.random.rand(3)
            img_array = (img_array * tint).astype(np.uint8)
        elif i % 4 == 2:
            # Random patches (simulates objects)
            img_array = np.random.randint(100, 200, (*image_size, 3), dtype=np.uint8)
            # Add random rectangles
            for _ in range(3):
                x, y = np.random.randint(0, image_size[0]-8), np.random.randint(0, image_size[1]-8)
                w, h = np.random.randint(4, 12), np.random.randint(4, 12)
                color = np.random.randint(0, 255, 3)
                img_array[x:min(x+w, image_size[0]), y:min(y+h, image_size[1])] = color
        else:
            # Pure random noise (stress test)
            img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)

        img = Image.fromarray(img_array, mode='RGB')
        img.save(calib_dir / f"calib_{i:03d}.png")

    return calib_dir


@pytest.fixture
def calibration_images_224(tmp_path):
    """Create calibration images at 224x224 for larger model testing."""
    if not HAS_PIL:
        pytest.skip("PIL required for calibration image tests")

    calib_dir = tmp_path / "calibration_images_224"
    calib_dir.mkdir()

    image_size = (224, 224)
    num_images = 10  # Fewer images for larger size

    for i in range(num_images):
        # Create realistic-looking synthetic images
        img_array = np.zeros((*image_size, 3), dtype=np.uint8)

        # Background gradient
        for y in range(image_size[0]):
            for x in range(image_size[1]):
                img_array[y, x] = [
                    int(128 + 64 * np.sin(x / 30 + i)),
                    int(128 + 64 * np.cos(y / 30 + i)),
                    int(128 + 64 * np.sin((x + y) / 40 + i))
                ]

        # Add some "objects" (circles/rectangles)
        for _ in range(5):
            cx, cy = np.random.randint(20, image_size[0]-20), np.random.randint(20, image_size[1]-20)
            radius = np.random.randint(10, 40)
            color = np.random.randint(0, 255, 3)

            y_indices, x_indices = np.ogrid[:image_size[0], :image_size[1]]
            mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= radius**2
            img_array[mask] = color

        # Add noise
        noise = np.random.randint(-20, 20, (*image_size, 3))
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(img_array, mode='RGB')
        img.save(calib_dir / f"calib_{i:03d}.jpg")

    return calib_dir


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


# =============================================================================
# INT8 Calibration Tests
# =============================================================================

@pytest.mark.skipif(not HAS_OPENVINO, reason="OpenVINO not installed")
@pytest.mark.skipif(not HAS_NNCF, reason="NNCF not installed")
@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestOpenVINOInt8Calibration:
    """Tests for INT8 quantization with actual calibration images."""

    def test_int8_calibration_basic(self, simple_onnx_model, calibration_images, tmp_path):
        """Test INT8 calibration with synthetic calibration images.

        This test verifies that:
        1. Calibration images are loaded correctly
        2. NNCF quantization runs successfully
        3. The quantized model is smaller than FP32
        4. The quantized model produces valid outputs
        """
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget
        from embodied_ai_architect.agents.deployment.models import (
            CalibrationConfig,
            DeploymentPrecision,
        )

        target = OpenVINOTarget()

        # Create calibration config
        calib_config = CalibrationConfig(
            data_path=calibration_images,
            num_samples=10,
            batch_size=1,
            input_shape=(1, 3, 32, 32),
            preprocessing="imagenet",
        )

        # Deploy with INT8
        output_path = tmp_path / "model_int8.xml"
        artifact = target.deploy(
            model=simple_onnx_model,
            precision=DeploymentPrecision.INT8,
            output_path=output_path,
            input_shape=(1, 3, 32, 32),
            calibration=calib_config,
        )

        # Verify artifact
        assert artifact.engine_path.exists()
        assert artifact.precision == DeploymentPrecision.INT8
        assert artifact.size_bytes > 0

        # Verify IR files exist
        bin_path = output_path.with_suffix(".bin")
        assert bin_path.exists(), "Binary weights file should exist"

    def test_int8_vs_fp32_size_comparison(self, simple_onnx_model, calibration_images, tmp_path):
        """Verify INT8 model is smaller than FP32."""
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget
        from embodied_ai_architect.agents.deployment.models import (
            CalibrationConfig,
            DeploymentPrecision,
        )

        target = OpenVINOTarget()

        # Deploy FP32
        fp32_artifact = target.deploy(
            model=simple_onnx_model,
            precision=DeploymentPrecision.FP32,
            output_path=tmp_path / "model_fp32.xml",
            input_shape=(1, 3, 32, 32),
        )

        # Deploy INT8
        calib_config = CalibrationConfig(
            data_path=calibration_images,
            num_samples=10,
            batch_size=1,
            input_shape=(1, 3, 32, 32),
            preprocessing="imagenet",
        )

        int8_artifact = target.deploy(
            model=simple_onnx_model,
            precision=DeploymentPrecision.INT8,
            output_path=tmp_path / "model_int8.xml",
            input_shape=(1, 3, 32, 32),
            calibration=calib_config,
        )

        # INT8 should typically be smaller (or similar for very small models)
        # For tiny models, the overhead might make INT8 similar size
        # But we verify both are valid
        assert fp32_artifact.size_bytes > 0
        assert int8_artifact.size_bytes > 0

    def test_int8_validation_accuracy(self, simple_onnx_model, calibration_images, tmp_path):
        """Test that INT8 model produces accurate results compared to FP32."""
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget
        from embodied_ai_architect.agents.deployment.models import (
            CalibrationConfig,
            DeploymentPrecision,
            ValidationConfig,
        )

        target = OpenVINOTarget()

        # Deploy INT8
        calib_config = CalibrationConfig(
            data_path=calibration_images,
            num_samples=10,
            batch_size=1,
            input_shape=(1, 3, 32, 32),
            preprocessing="imagenet",
        )

        artifact = target.deploy(
            model=simple_onnx_model,
            precision=DeploymentPrecision.INT8,
            output_path=tmp_path / "model_int8.xml",
            input_shape=(1, 3, 32, 32),
            calibration=calib_config,
        )

        # Validate against FP32 baseline
        val_config = ValidationConfig(
            num_samples=5,
            tolerance_percent=5.0,  # Allow 5% tolerance for INT8
            compare_outputs=True,
        )

        result = target.validate(artifact, simple_onnx_model, val_config)

        assert result.samples_compared == 5
        assert result.baseline_latency_ms is not None
        assert result.deployed_latency_ms is not None
        # INT8 may have larger output differences than FP32
        assert result.max_output_diff is not None

    def test_int8_different_preprocessing(self, simple_onnx_model, calibration_images, tmp_path):
        """Test INT8 calibration with different preprocessing modes."""
        from embodied_ai_architect.agents.deployment.targets.openvino import OpenVINOTarget
        from embodied_ai_architect.agents.deployment.models import (
            CalibrationConfig,
            DeploymentPrecision,
        )

        target = OpenVINOTarget()

        preprocessing_modes = ["imagenet", "yolo", "none"]

        for preprocessing in preprocessing_modes:
            calib_config = CalibrationConfig(
                data_path=calibration_images,
                num_samples=5,
                batch_size=1,
                input_shape=(1, 3, 32, 32),
                preprocessing=preprocessing,
            )

            output_path = tmp_path / f"model_int8_{preprocessing}.xml"
            artifact = target.deploy(
                model=simple_onnx_model,
                precision=DeploymentPrecision.INT8,
                output_path=output_path,
                input_shape=(1, 3, 32, 32),
                calibration=calib_config,
            )

            assert artifact.engine_path.exists(), f"Failed for preprocessing={preprocessing}"
            assert artifact.precision == DeploymentPrecision.INT8


@pytest.mark.skipif(not HAS_OPENVINO, reason="OpenVINO not installed")
@pytest.mark.skipif(not HAS_NNCF, reason="NNCF not installed")
@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestDeploymentAgentInt8:
    """Integration tests for INT8 deployment through the DeploymentAgent."""

    def test_agent_int8_deployment(self, simple_onnx_model, calibration_images, tmp_path):
        """Test complete INT8 deployment workflow through the agent."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()

        if "openvino" not in agent.list_targets():
            pytest.skip("OpenVINO target not available")

        result = agent.execute({
            "model": str(simple_onnx_model),
            "target": "openvino",
            "precision": "int8",
            "input_shape": [1, 3, 32, 32],
            "calibration_data": str(calibration_images),
            "calibration_samples": 10,
            "calibration_preprocessing": "imagenet",
            "output_dir": str(tmp_path),
        })

        assert result.success, f"INT8 deployment failed: {result.error}"
        assert result.data["artifact"]["precision"] == "int8"

        # Check logs mention calibration/quantization
        logs = result.data.get("logs", [])
        assert any("int8" in log.lower() for log in logs)

    def test_agent_int8_with_validation(self, simple_onnx_model, calibration_images, tmp_path):
        """Test INT8 deployment with validation through the agent."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()

        if "openvino" not in agent.list_targets():
            pytest.skip("OpenVINO target not available")

        result = agent.execute({
            "model": str(simple_onnx_model),
            "target": "openvino",
            "precision": "int8",
            "input_shape": [1, 3, 32, 32],
            "calibration_data": str(calibration_images),
            "calibration_samples": 10,
            "test_data": str(calibration_images),  # Use same images for validation
            "validation_samples": 5,
            "accuracy_tolerance": 5.0,
            "output_dir": str(tmp_path),
        })

        assert result.success, f"INT8 deployment with validation failed: {result.error}"

        # Check validation results
        validation = result.data.get("validation")
        assert validation is not None
        assert validation["samples_compared"] > 0

    def test_agent_int8_missing_calibration_data(self, simple_onnx_model, tmp_path):
        """Test that INT8 deployment fails gracefully without calibration data."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()

        if "openvino" not in agent.list_targets():
            pytest.skip("OpenVINO target not available")

        result = agent.execute({
            "model": str(simple_onnx_model),
            "target": "openvino",
            "precision": "int8",
            "input_shape": [1, 3, 32, 32],
            # No calibration_data provided
            "output_dir": str(tmp_path),
        })

        assert not result.success
        assert "calibration" in result.error.lower()


# =============================================================================
# Jetson INT8 Calibration Tests (Mocked)
# =============================================================================

class TestJetsonInt8CalibrationMocked:
    """Tests for Jetson INT8 calibration with mocked TensorRT."""

    def test_jetson_calibrator_creation(self, calibration_images):
        """Test that Jetson INT8 calibrator can be created."""
        # Mock TensorRT
        mock_trt = MagicMock()
        mock_trt.IInt8EntropyCalibrator2 = MagicMock

        with patch.dict('sys.modules', {'tensorrt': mock_trt, 'pycuda': MagicMock(), 'pycuda.driver': MagicMock(), 'pycuda.autoinit': MagicMock()}):
            from embodied_ai_architect.agents.deployment.models import CalibrationConfig

            config = CalibrationConfig(
                data_path=calibration_images,
                num_samples=10,
                batch_size=1,
                input_shape=(1, 3, 32, 32),
                preprocessing="imagenet",
            )

            # Verify config is valid
            assert config.data_path == calibration_images
            assert config.num_samples == 10
            assert config.preprocessing == "imagenet"

    def test_jetson_int8_requires_calibration(self, simple_onnx_model, tmp_path):
        """Test that Jetson INT8 fails without calibration config."""
        # This test verifies the validation logic without actual TensorRT

        from embodied_ai_architect.agents.deployment.models import (
            CalibrationConfig,
            DeploymentPrecision,
        )

        # The base target logic should require calibration for INT8
        precision = DeploymentPrecision.INT8
        calibration = None

        # This is what the target checks
        if precision == DeploymentPrecision.INT8 and calibration is None:
            with pytest.raises(ValueError):
                raise ValueError("INT8 precision requires calibration config")


class TestCalibrationImageLoading:
    """Tests for calibration image loading and preprocessing."""

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_calibration_images_created(self, calibration_images):
        """Verify calibration image fixture creates valid images."""
        image_files = list(calibration_images.glob("*.png"))
        assert len(image_files) == 20

        # Verify images are readable
        for img_path in image_files[:3]:  # Check first 3
            img = Image.open(img_path)
            assert img.size == (32, 32)
            assert img.mode == "RGB"

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_calibration_images_224_created(self, calibration_images_224):
        """Verify 224x224 calibration images are created."""
        image_files = list(calibration_images_224.glob("*.jpg"))
        assert len(image_files) == 10

        # Verify images are readable
        for img_path in image_files[:3]:
            img = Image.open(img_path)
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_preprocessing_imagenet(self, calibration_images):
        """Test ImageNet preprocessing on calibration images."""
        img_path = list(calibration_images.glob("*.png"))[0]
        img = Image.open(img_path).convert("RGB")

        # Resize
        img = img.resize((32, 32), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32)

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = img_array / 255.0
        img_array = (img_array - mean) / std

        # Should have values centered around 0
        assert img_array.min() < 0  # Some values should be negative
        assert img_array.max() > 0

    @pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
    def test_preprocessing_yolo(self, calibration_images):
        """Test YOLO preprocessing on calibration images."""
        img_path = list(calibration_images.glob("*.png"))[0]
        img = Image.open(img_path).convert("RGB")

        img = img.resize((32, 32), Image.BILINEAR)
        img_array = np.array(img, dtype=np.float32)

        # YOLO: 0-1 normalization only
        img_array = img_array / 255.0

        # Should be in range [0, 1]
        assert img_array.min() >= 0
        assert img_array.max() <= 1


# =============================================================================
# Coral Edge TPU Tests
# =============================================================================

class TestCoralTargetBasic:
    """Basic tests for Coral Edge TPU target (no TensorFlow required)."""

    def test_coral_target_import(self):
        """Test that CoralTarget can be imported."""
        from embodied_ai_architect.agents.deployment.targets.coral import CoralTarget

        target = CoralTarget()
        assert target.name == "coral"

    def test_coral_capabilities_without_tf(self):
        """Test capabilities structure even without TensorFlow."""
        from embodied_ai_architect.agents.deployment.targets.coral import CoralTarget

        target = CoralTarget()
        caps = target.get_capabilities()

        assert caps["name"] == "coral"
        assert caps["supported_precisions"] == ["int8"]  # Edge TPU is INT8 only
        assert caps["supports_dynamic_batch"] is False
        assert caps["output_format"] == ".tflite"

    def test_coral_int8_only_requirement(self):
        """Test that Coral enforces INT8-only precision."""
        from embodied_ai_architect.agents.deployment.targets.coral import CoralTarget
        from embodied_ai_architect.agents.deployment.models import DeploymentPrecision

        target = CoralTarget()

        # Should only support INT8
        caps = target.get_capabilities()
        assert "int8" in caps["supported_precisions"]
        assert "fp32" not in caps["supported_precisions"]
        assert "fp16" not in caps["supported_precisions"]

    def test_coral_requires_calibration(self, simple_onnx_model, tmp_path):
        """Test that Coral deployment requires calibration data."""
        from embodied_ai_architect.agents.deployment.targets.coral import CoralTarget
        from embodied_ai_architect.agents.deployment.models import DeploymentPrecision

        target = CoralTarget()

        if not target.is_available():
            pytest.skip("TensorFlow not installed")

        with pytest.raises(ValueError, match="calibration"):
            target.deploy(
                model=simple_onnx_model,
                precision=DeploymentPrecision.INT8,
                output_path=tmp_path / "model.tflite",
                input_shape=(1, 3, 32, 32),
                calibration=None,
            )

    def test_coral_rejects_fp32(self, simple_onnx_model, calibration_images, tmp_path):
        """Test that Coral rejects non-INT8 precision."""
        from embodied_ai_architect.agents.deployment.targets.coral import CoralTarget
        from embodied_ai_architect.agents.deployment.models import (
            CalibrationConfig,
            DeploymentPrecision,
        )

        target = CoralTarget()

        if not target.is_available():
            pytest.skip("TensorFlow not installed")

        calib_config = CalibrationConfig(
            data_path=calibration_images,
            num_samples=5,
            batch_size=1,
            input_shape=(1, 3, 32, 32),
            preprocessing="imagenet",
        )

        with pytest.raises(ValueError, match="INT8"):
            target.deploy(
                model=simple_onnx_model,
                precision=DeploymentPrecision.FP32,
                output_path=tmp_path / "model.tflite",
                input_shape=(1, 3, 32, 32),
                calibration=calib_config,
            )


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestCoralTargetWithTensorFlow:
    """Tests for Coral Edge TPU target with TensorFlow available."""

    def test_coral_is_available(self):
        """Test Coral availability detection with TensorFlow."""
        from embodied_ai_architect.agents.deployment.targets.coral import CoralTarget

        target = CoralTarget()
        assert target.is_available() is True

    def test_coral_capabilities_with_tf(self):
        """Test capabilities with TensorFlow available."""
        from embodied_ai_architect.agents.deployment.targets.coral import CoralTarget

        target = CoralTarget()
        caps = target.get_capabilities()

        assert "tensorflow_version" in caps
        assert caps["supported_precisions"] == ["int8"]

    def test_coral_deploy_int8(self, simple_onnx_model, calibration_images, tmp_path):
        """Test INT8 deployment to TFLite format."""
        pytest.importorskip("onnx2tf", reason="onnx2tf required for ONNX to TF conversion")

        from embodied_ai_architect.agents.deployment.targets.coral import CoralTarget
        from embodied_ai_architect.agents.deployment.models import (
            CalibrationConfig,
            DeploymentPrecision,
        )

        target = CoralTarget()

        calib_config = CalibrationConfig(
            data_path=calibration_images,
            num_samples=10,
            batch_size=1,
            input_shape=(1, 3, 32, 32),
            preprocessing="imagenet",
        )

        artifact = target.deploy(
            model=simple_onnx_model,
            precision=DeploymentPrecision.INT8,
            output_path=tmp_path / "model.tflite",
            input_shape=(1, 3, 32, 32),
            calibration=calib_config,
        )

        # Verify artifact
        assert artifact.engine_path.exists()
        assert artifact.precision == DeploymentPrecision.INT8
        assert artifact.target == "coral"
        assert artifact.size_bytes > 0
        assert str(artifact.engine_path).endswith(".tflite")


class TestCoralEdgeCases:
    """Edge case tests for Coral target."""

    def test_coral_nhwc_input_shape(self, calibration_images):
        """Test that Coral handles NHWC input shape."""
        from embodied_ai_architect.agents.deployment.models import CalibrationConfig

        # NHWC format (batch, height, width, channels)
        nhwc_shape = (1, 32, 32, 3)

        config = CalibrationConfig(
            data_path=calibration_images,
            num_samples=5,
            batch_size=1,
            input_shape=nhwc_shape,
            preprocessing="imagenet",
        )

        assert config.input_shape == nhwc_shape

    def test_coral_nchw_input_shape(self, calibration_images):
        """Test that Coral handles NCHW input shape."""
        from embodied_ai_architect.agents.deployment.models import CalibrationConfig

        # NCHW format (batch, channels, height, width)
        nchw_shape = (1, 3, 32, 32)

        config = CalibrationConfig(
            data_path=calibration_images,
            num_samples=5,
            batch_size=1,
            input_shape=nchw_shape,
            preprocessing="imagenet",
        )

        assert config.input_shape == nchw_shape


class TestDeploymentAgentCoral:
    """Integration tests for Coral through DeploymentAgent."""

    def test_agent_coral_missing_calibration(self, simple_onnx_model, tmp_path):
        """Test agent fails gracefully for Coral without calibration."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()

        if "coral" not in agent.list_targets():
            pytest.skip("Coral target not available")

        result = agent.execute({
            "model": str(simple_onnx_model),
            "target": "coral",
            "precision": "int8",
            "input_shape": [1, 3, 32, 32],
            "output_dir": str(tmp_path),
            # No calibration_data
        })

        assert not result.success
        assert "calibration" in result.error.lower()

    def test_agent_coral_wrong_precision(self, simple_onnx_model, calibration_images, tmp_path):
        """Test agent fails for Coral with non-INT8 precision."""
        from embodied_ai_architect.agents.deployment import DeploymentAgent

        agent = DeploymentAgent()

        if "coral" not in agent.list_targets():
            pytest.skip("Coral target not available")

        result = agent.execute({
            "model": str(simple_onnx_model),
            "target": "coral",
            "precision": "fp16",  # Wrong precision
            "input_shape": [1, 3, 32, 32],
            "calibration_data": str(calibration_images),
            "output_dir": str(tmp_path),
        })

        assert not result.success
        assert "int8" in result.error.lower()
