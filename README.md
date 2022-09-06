[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8320448&assignment_repo_type=AssignmentRepo)
# EMLO V2 - Session 01

## 01 - Docker

## Steps:

#### Created and tested in gitpod - https://www.gitpod.io/ 

1. Create Dockerfile that uses https://github.com/rwightman/pytorch-image-models (timm models) with entrypoint to inference.py taking arguments("$@")
2. Create inference.py file for timm model with configurable parameters (model (resnet, efficient_b0), image(testimageurl)), the output needs to be a json object in " " for all the testcases (all_tests.py) to pass.
3. Update the requirements.txt file with needed package timm and the ** link** to download torch and torchvision wheel files https://download.pytorch.org/whl/torch_stable.html 
4. Build docker image using docker build --tag <imagename>.
5. Use docker images to see the image size.
6. Usage of alpine or slim images - used slim here to reduce size. 
7. *To reduce the image size, use the CPU version of torch & torchvision packages. (reduces to 1.09GB from 2.5GB) *
8. Build docker image using docker build --rm -it <imagename> --model resnet18 image https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg and pass the model and image parameters.
9. The docker image prints output "{"predicted": "samoyed", "confidence": "0.88"}" in JSON format.
10. Push the docker image to dockerhub
11. The code needs to pass all testcases in all_tests.py. bash test/all_tests.py
  
  
## Run
- docker pull premavasagam/mlops:s1_docker
- docker run --rm -it premavasagam/mlops:s1_docker --model resnet18 --image https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
- docker images (lists all docker images with its size)


