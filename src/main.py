from torchvision import models

if __name__ == "__main__":
    model = models.squeezenet1_0(pretrained=True)
