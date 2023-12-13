#!/bin/bash

# Login to AWS ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 260939427961.dkr.ecr.us-west-2.amazonaws.com

# Build Docker Image
docker build -t pulse-backend:3.07 .

# Tag Docker Image
docker tag pulse-backend:3.07 260939427961.dkr.ecr.us-west-2.amazonaws.com/pulse-backend:3.07

# Push Docker Image to ECR
docker push 260939427961.dkr.ecr.us-west-2.amazonaws.com/pulse-backend:3.07

# Git operations
git init
git add .
git commit -m "New deployment"
git push -u origin master

# AWS EB operations
eb use flask-production
eb deploy
