variable "aws_region" {
  description = "AWS region for the environment."
  type        = string
  default     = "eu-north-1"
}

variable "project_name" {
  description = "Lowercase project identifier used in resource names."
  type        = string
  default     = "zephyrwerk"
}

variable "environment" {
  description = "Deployment environment identifier."
  type        = string
  default     = "dev"
}

variable "db_name" {
  description = "Initial PostgreSQL database name."
  type        = string
  default     = "zephyrwerk"
}

variable "db_username" {
  description = "PostgreSQL master username."
  type        = string
  default     = "zephyrwerk_admin"
}

variable "db_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage_gb" {
  description = "Initial gp3 storage size for RDS in GiB."
  type        = number
  default     = 20
}

variable "db_backup_retention_days" {
  description = "Number of daily automated RDS backups to retain."
  type        = number
  default     = 7
}