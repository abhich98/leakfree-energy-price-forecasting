# Infrastructure Baseline

This Terraform configuration creates one low-cost development environment in `eu-central-1`:

- a private, encrypted, versioned S3 bucket for data and artifacts;
- a VPC with two public and two private subnets;
- a private single-AZ PostgreSQL 16 RDS instance (`db.t4g.micro`, 20 GiB gp3);
- a Secrets Manager secret managed by RDS for the database master password; and
- security groups ready for future Fargate and Lambda workloads.

It uses local Terraform state. State files are ignored by Git and must not be committed. Before automated GitHub Actions applies are enabled, migrate the state to a dedicated, private S3 backend with locking.

## Prerequisites

- Terraform 1.6 or newer
- AWS CLI credentials for an AWS account allowed to create the listed resources in `eu-central-1`

## Apply

```bash
cd infra
terraform init
terraform plan
terraform apply
```

Terraform will show the resources and estimated changes before asking for confirmation. This creates billable AWS resources, especially RDS.

## Retrieve connection settings

```bash
terraform output data_bucket_name
terraform output database_endpoint
terraform output -raw database_secret_arn
```

The database remains private. Do not make it public to connect from a laptop. A future Fargate task or Lambda function in this VPC will retrieve the RDS-managed secret and connect through a dedicated compute security group.

## Teardown

When the environment is no longer needed:

```bash
terraform destroy
```

S3 bucket deletion will fail until it is empty, including noncurrent versions. This is intentional protection for versioned data.

## Basics

The `infra` directory contains these Terraform files:

- `versions.tf`: declares the Terraform and AWS provider versions the project supports.
- `variables.tf`: defines configurable inputs, such as region, database name, and RDS size.
- `main.tf`: defines the AWS resources Terraform should create: S3, networking, security groups, and RDS.
- `outputs.tf`: prints useful values after an apply, such as the bucket name and private database endpoint.
- `.terraform.lock.hcl`: records the exact provider version Terraform selected. Commit this file so every machine uses the same provider version.

After `terraform apply`, Terraform also creates `terraform.tfstate` locally. It records the IDs and current configuration of the AWS resources Terraform manages. It is ignored by Git because it can contain sensitive details. Do not edit it manually or commit it.
