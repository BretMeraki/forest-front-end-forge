import pytest
from forest_app.integrations.prompt_augmentation import PromptAugmentationService, AugmentationTemplate

def test_default_template_exists():
    service = PromptAugmentationService()
    assert "json_generation" in service.templates
    template = service.templates["json_generation"]
    assert isinstance(template, AugmentationTemplate)

def test_format_prompt_with_examples():
    template = AugmentationTemplate(
        name="test",
        description="Test template",
        system_prompt="System context.",
        prompt_format="Hello, {name}!",
        examples=[{"input": "Hi", "output": "Hello!"}]
    )
    result = template.format_prompt(name="World")
    assert isinstance(result, list)
    assert result[0]["role"] == "system"
    assert result[-1]["content"] == "Hello, World!"

def test_format_prompt_missing_param():
    template = AugmentationTemplate(
        name="test",
        description="Test template",
        system_prompt="System context.",
        prompt_format="Hello, {name}!"
    )
    with pytest.raises(ValueError):
        template.format_prompt()
