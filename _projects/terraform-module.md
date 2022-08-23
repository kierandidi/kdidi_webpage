---
layout: project
title: 'Terraform Module for EMR Serverless'
caption: IaC Template for Big Data Jobs on AWS
description: >
  A template to create your own EMR Serverless cluster on AWS, including all neccessary IAM roles and some extra features.
date: '19-08-2022'
image: 
  path: /assets/img/projects/cloud_logo.jpg
#links:
#  - title: Link
#    url: https://hydejack.com/
sitemap: false
---

# Terraform Module for EMR Serverless
We recently published a [Terraform module for EMR Serverless](https://registry.terraform.io/modules/kierandidi/emrserverless/aws/1.0.0) that came out of my internship at the [translational bioinformatics group](https://www.bioinformatics.csiro.au/) at CSIRO in Sydney.

With this module, we want to make it easier to use this relatively new service at AWS (released in June 2022) which we think is really useful for scalable and cost-effective big data analysis. While the README document on the Terraform Registry is aimed to be short and concise and the [AWS Documentation](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html) can be quite overwhelming at first encounter, here I will explain some of the aspects of working with our module and EMR Serverless in general in a bit more detail.


* toc
{:toc}

## Why a Terraform module

In my internship I was building a cloud-native bioinformatics software for genomic analysis (see [this blog post]() if you want to know more about the context). We chose to use the EMR Serverless service of AWS since it allowed us to run our Spark application in a transient cluster that gets shut-down after job completion, ideal for our pipeline with fluctuating workloads.

Since the service was just published in June 2022, there was no Infrastructure-as-Code solution publicly available. Nevertheless, we wanted to use IaC in order to automate the provisioning process and remove error potential at this part of the pipeline in order to fully focus on the development of the application code. 

That is why we created the Terraform module: It takes care of all the IAM roles, bucket policies etc. and even adds some nice features such as uploading your virtual environments and your source code to the cloud.
## How to use our Terraform module

After installing Terraform, the process is quite straightforward: You create a main.tf file in your working directory in which you also locate the compressed virtual environment and your source code folder (see the [usage notes](https://registry.terraform.io/modules/kierandidi/emrserverless/aws/1.0.0)) with content similar to the following: 

~~~

~~~
Then, you can follow the typical Terraform workflow: 

1. run `terraform init` to download and install the necessary providers.
2. run `terraform plan` to see what resources will be provisioned upon execution
3. run `terraform apply` to execute the proposed resource provisioning plan (you can add `--auto-approve` if you do not want to type `yes` for every apply). After running it, you should see an output with some useful information, such as the application id of your created application and the path to your uploaded source code (it will have a different name now since its name is based on a hash; that helps the module figuring out if there were any changes between this and the last upload).
4. in case you want to destroy the infrastructure at some point including associated IAM roles etc, just run `terraform destroy`.

## What to do after Terraform apply?

Once you have executed `terraform apply`, the infrastructure for your code is fully provisioned, and there is just one step left to submit your first job run.

First of all, you need a job script to run. With our module you can directly upload the environment (venv/conda) and the source code needed for the application, but we decided not to include the job script in the provisioning script since this can change quite often and is also not directly part of the infrastructure.

You can upload your script either via the Management Console or via the AWS CLI in your terminal using the following command: 

`aws s3 cp <filename>  s3://<bucket_name>/`

Optionally you can add folders after `s3://<bucket_name>/` if you do want your job file to reside in a specific folder.

## Submit your first job

Submitting your jobs can (as often) either be done in the management console or in the terminal via the AWS CLI. What I like about the console option is that you can easily clone jobs; this creates a template for a new job including all the parameters of the old job, saving you the time to retype everything and prevents you from making mistakes. 

Here is a screenshot of what those parameters might look like for a typical application: 



For the CLI option, you can use the following command: 


~~~
aws emr-serverless start-job-run \
    --application-id <application-id> \
    --execution-role-arn <execution-role-arn> \
    --name <application_name> \
    --job-driver '{
        "sparkSubmit": {
          "entryPoint": "<s3location_of_job_file>",
          "entryPointArguments": [<argument1>, <argument2>, <...>],
          "sparkSubmitParameters": "--conf spark.executor.cores=1 --conf spark.executor.memory=4g --conf spark.driver.cores=1 --conf spark.driver.memory=4g --conf spark.executor.instances=1"
        }
    }'
~~~

The `sparkSubmitParameters` are just an example. The most important bit are the entryPoint and the entryPointArguments, so the file you want to run on the cluster and the arguments you want to pass to it.

You can configure an awful lot when submitting a job. One of the options that I find quite useful is determining a location in your S3 bucket where the log files of the jobs are stored; they help immensely in case of troubleshooting. 

To do that, you can tick the corresponding box in the job submissions form in the management console or provide the following `--configuration-overrides` besides the other configurations above: 

~~~
{
    "monitoringConfiguration": {
        "s3MonitoringConfiguration": {
            "logUri": "s3://<your-s3bucket>/logs/"
        }
    }
}
~~~

It will create a log folder which then nicely sorts the logs first based on application id, then job id and the Spark Driver/Executor so that you do not drown in a sea full of log files (not a nice feeling I can assure you).
## Monitoring your running jobs

You can use the console to monitor your applications and jobs. The only time I use this frequently is to see how a job is going without leaving the terminal, but you can do a lot more with it in case you want to.

~~~
aws emr-serverless get-job-run \
    --application-id <application-id> \
    --job-run-id <job-run-id>
~~~

You get the `job-run-id` as output in your terminal if you submit a job via the CLI or you can see it in the management console under the details of your job.