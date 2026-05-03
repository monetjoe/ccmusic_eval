import torch
import torch.nn as nn
import torchvision.models as models
from utils import MODEL_DIR, EN_US


class EvalNet:
    model: nn.Module = None
    m_type = "squeezenet"
    input_size = 224
    output_size = 512

    def __init__(self, log_name: str, cls_num: int):
        saved_model_path = f"{MODEL_DIR}/{log_name}/save.pt"
        m_ver = "_".join(log_name.split("_")[:-3])
        self.m_type, self.input_size = self._model_info(m_ver)

        if not hasattr(models, m_ver):
            raise ValueError("不支持的模型")

        self.model = eval("models.%s()" % m_ver)
        linear_output = self._set_outsize()
        self._set_classifier(cls_num, linear_output)
        checkpoint = torch.load(
            saved_model_path,
            map_location=torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
        )
        self.model.load_state_dict(checkpoint, False)
        self.model.eval()

    def _get_backbone(self, ver: str, backbone_list: list):
        for bb in backbone_list:
            if ver == bb["ver"]:
                return bb

        print("未找到骨干网络名称，使用默认选项 - alexnet。")
        return backbone_list[0]

    def _model_info(self, m_ver: str):
        if EN_US:
            from datasets import load_dataset

            backbone_list = load_dataset("monetjoe/cv_backbones", split="train")

        else:
            from modelscope.msdatasets import MsDataset

            backbone_list = MsDataset.load("monetjoe/cv_backbones", split="train")

        backbone = self._get_backbone(m_ver, backbone_list)
        m_type = str(backbone["type"])
        input_size = int(backbone["input_size"])
        return m_type, input_size

    def _classifier(self, cls_num: int, output_size: int, linear_output: bool):
        q = (1.0 * output_size / cls_num) ** 0.25
        l1 = int(q * cls_num)
        l2 = int(q * l1)
        l3 = int(q * l2)
        if linear_output:
            return torch.nn.Sequential(
                nn.Dropout(),
                nn.Linear(output_size, l3),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(l3, l2),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(l2, l1),
                nn.ReLU(inplace=True),
                nn.Linear(l1, cls_num),
            )

        else:
            return torch.nn.Sequential(
                nn.Dropout(),
                nn.Conv2d(output_size, l3, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(l3, l2),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(l2, l1),
                nn.ReLU(inplace=True),
                nn.Linear(l1, cls_num),
            )

    def _set_outsize(self):
        for name, module in self.model.named_modules():
            if (
                str(name).__contains__("classifier")
                or str(name).__eq__("fc")
                or str(name).__contains__("head")
                or hasattr(module, "classifier")
            ):
                if isinstance(module, torch.nn.Linear):
                    self.output_size = module.in_features
                    return True

                if isinstance(module, torch.nn.Conv2d):
                    self.output_size = module.in_channels
                    return False

        return False

    def _set_classifier(self, cls_num: int, linear_output: bool):
        if self.m_type == "convnext":
            del self.model.classifier[2]
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier)
                + list(self._classifier(cls_num, self.output_size, linear_output))
            )
            return

        elif self.m_type == "maxvit":
            del self.model.classifier[5]
            self.model.classifier = nn.Sequential(
                *list(self.model.classifier)
                + list(self._classifier(cls_num, self.output_size, linear_output))
            )
            return

        if hasattr(self.model, "classifier"):
            self.model.classifier = self._classifier(
                cls_num, self.output_size, linear_output
            )
            return

        elif hasattr(self.model, "fc"):
            self.model.fc = self._classifier(cls_num, self.output_size, linear_output)
            return

        elif hasattr(self.model, "head"):
            self.model.head = self._classifier(cls_num, self.output_size, linear_output)
            return

        self.model.heads.head = self._classifier(
            cls_num, self.output_size, linear_output
        )

    def forward(self, x: torch.Tensor):
        if torch.cuda.is_available():
            x = x.cuda()
            self.model = self.model.cuda()

        if self.m_type == "googlenet":
            return self.model(x)[0]
        else:
            return self.model(x)
