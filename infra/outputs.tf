output "aws_region" {
  description = "AWS region for this environment."
  value       = var.aws_region
}

output "data_bucket_name" {
  description = "Private, versioned S3 bucket for data and artifacts."
  value       = aws_s3_bucket.data_lake.bucket
}

output "database_endpoint" {
  description = "Private PostgreSQL endpoint, reachable only from approved VPC security groups."
  value       = aws_db_instance.main.address
}

output "database_port" {
  description = "PostgreSQL port."
  value       = aws_db_instance.main.port
}

output "database_secret_arn" {
  description = "Secrets Manager ARN containing the RDS-managed master credentials."
  value       = aws_db_instance.main.master_user_secret[0].secret_arn
  sensitive   = true
}

output "vpc_id" {
  description = "VPC identifier for future Fargate and Lambda networking."
  value       = aws_vpc.main.id
}

output "private_subnet_ids" {
  description = "Private subnet IDs used by RDS and future VPC-attached compute."
  value       = [for subnet in aws_subnet.private : subnet.id]
}

output "database_security_group_id" {
  description = "Attach future Fargate/Lambda security groups as ingress sources on this group."
  value       = aws_security_group.database.id
}