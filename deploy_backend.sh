#!/bin/bash

# Login to AWS ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 260939427961.dkr.ecr.us-west-2.amazonaws.com

# Build Docker Image
docker build -t pulse-backend .

# Tag Docker Image
docker tag pulse-backend:latest 260939427961.dkr.ecr.us-west-2.amazonaws.com/pulse-backend:latest

# Push Docker Image to ECR
docker push 260939427961.dkr.ecr.us-west-2.amazonaws.com/pulse-backend:latest

# Git operations
git init
git add .
git commit -m "new push"
git push -u origin master

# AWS EB operations
eb use flask-production
eb deploy
