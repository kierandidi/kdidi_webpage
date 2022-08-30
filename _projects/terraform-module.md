---
layout: project
title: 'Terraform Module for EMR Serverless'
caption: IaC Template for Big Data Jobs on AWS
description: >
  A template to create your own EMR Serverless cluster on AWS, including all necessary IAM roles and some extra features.
date: '19-08-2022'
image: 
  path: /assets/img/projects/cloud_logo.jpg
#links:
#  - title: Link
#    url: https://hydejack.com/
sitemap: false
---

# Terraform Module for EMR Serverless
Together with [Anuradha Wickramarachchi](https://anuradhawick.com/) I recently published a [Terraform module for EMR Serverless](https://registry.terraform.io/modules/kierandidi/emrserverless/aws/1.0.0) that came out of my internship at the [Transformational Bioinformatics Group](https://www.bioinformatics.csiro.au/) at CSIRO in Sydney.

With this module, we want to make it easier to use this relatively new service at AWS (released in June 2022) which we think is really useful for scalable and cost-effective big data analysis. While the README document on the Terraform Registry is intended to be short and concise, the [AWS Documentation](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html) can be quite overwhelming at first sight. So here I decided to take the middleground: I will explain some of the aspects of working with our module and EMR Serverless in a bit more detail, focusing on the aspects I use often and without aiming to cover the full docs.


* toc
{:toc}

## Why a Terraform module

In my internship I was building a cloud-native bioinformatics software for genomic analysis (see [this blog post](https://bioinformatics.csiro.au/blog/the-many-ways-to-the-cloud-from-on-prem-to-emr-serverless/) if you want to know more about the context). We chose to use the EMR Serverless service of AWS since it allowed us to run our Spark application in a transient cluster that gets shut-down after job completion, ideal for our pipeline with fluctuating workloads.

Since the service was just published in June 2022, there was no Infrastructure-as-Code solution publicly available yet. Nevertheless, we wanted to use IaC in order to automate the provisioning process and remove error potential at this part of the pipeline in order to fully focus on the development of the application code. 

That is why we created the Terraform module: it takes care of all the IAM roles, bucket policies etc. and even adds some nice features such as uploading your virtual environments and your source code to the cloud.
## How to use our Terraform module

After installing Terraform, the process is quite straightforward: you create a main.tf file in your working directory, in which you also locate the compressed virtual environment and your source code folder (see the [usage notes](https://registry.terraform.io/modules/kierandidi/emrserverless/aws/1.0.0) or the paragraph below for more info) with content similar to the following: 

~~~terraform
# file: "main.tf"
module "emrserverless" {
  source  = "kierandidi/emrserverless/aws"
  version = "1.0.0"

  application_name       = "<name of application to be created>"
  scripts                = "<scripts folder>"
  env                    = "<compressed environment>"
  bucket_name            = "<name of bucket to be created>"
  application_max_memory = "8 GB"
  application_max_cores  = "1 vCPU"
}

output "emrserverless_outputs" {
  value = module.emrserverless
}
~~~
Then, you can follow the typical Terraform workflow: 

1. Run `terraform init` to download and install the necessary providers.
2. Run `terraform plan` to see what resources will be provisioned upon execution.
3. Run `terraform apply` to execute the proposed resource provisioning plan (you can add `--auto-approve` if you do not want to type `yes` for every apply). After running it, you should see an output with some useful information, such as the application id of your newly created application and the path to your uploaded source code (it will have a different name now since its name is based on a hash; that helps the module to figure out if there were any changes between this and the last upload).
4. In case you want to destroy the infrastructure at some point, including associated IAM roles etc, just run `terraform destroy`.

### Compress your source code and environment

In case you do not know how to compress your environment and script file, here are some short instructions:

- environment (conda):
  1. Install [conda pack](https://conda.github.io/conda-pack/) via `conda install conda-pack` or `pip3 install conda-pack`.
  2. Compress your conda environment via `conda pack -n <envname> -o <envname>.tar.gz`.
  3. Put it in the same directory as your `main.tf` file.

- environment (venv), also described in [AWS docs](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/using-python-libraries.html):
  1. Install [venv-pack](https://jcristharif.com/venv-pack/) via `pip3 install venv-pack`
  2. Compress the currently activate venv via `venv-pack -f -o <envname>.tar.gz`
  3. Put it in the same directory as your `main.tf` file.

- source code: 
  1. Define a bash function to only zip files you want (in my case I want to exclude .git and .DS_Store files since I am on a Mac): `function zip_clean() {zip -r $1 $2 -x "*.git*" -x "*.DS_Store"}`
  2. inside the directory you want to zip, execute the function: `zip_clean <folder-name>.zip .`
  3. Put it in the same directory as the `main.tf` file.

## What to do after Terraform apply?

Once you have executed `terraform apply`, the infrastructure for your code is fully provisioned, and there is just one step left to submit your first job run.

First of all, you need a job script to run. With our module you can directly upload the environment (venv/conda) and the source code needed for the application, but we decided not to include the job script in the provisioning script since this can change quite often and is also not directly part of the infrastructure.

You can upload your script either via the Management Console or via the AWS CLI in your terminal using the following command: 

`aws s3 cp <filename>  s3://<bucket_name>/`

Optionally you can add folders after `s3://<bucket_name>/` if you do want your job file to reside in a specific folder.

## Submit your first job

Submitting your jobs can (as usual) either be done in the management console or in the terminal via the AWS CLI. What I like about the console option is that you can easily clone jobs; this creates a template for a new job including all the parameters of the old job, saving you the time of retyping everything and preventing you from making mistakes. 

Here is a table of the typical parameters I set when running an EMR Serverless Spark job: 


| Name | Description | Value |
|------|-------------|---------------|
| <a name="spark.driver.cores"></a>[spark.driver.cores](#) | Number of cores available to Spark driver | `<number of cores>` |
| <a name="spark.driver.memory"></a>[spark.driver.memory](#) | Amount of memory available to Spark executor | `<memory in g, e.g. 4g>` |
| <a name="spark.executor.cores"></a>[spark.executor.cores](#) | Number of cores available to Spark driver  | `<number of cores>` |
| <a name="spark.executor.memory"></a>[spark.executor.memory](#) | Amount of memory available to Spark executor | `<memory in g, e.g. 4g>` |
| <a name="spark.archives"></a>[spark.archives](#) | Location of compressed environment | `s3://<path to your environment>#environment` |
| <a name="spark.submit.pyFiles"></a>[spark.submit.pyFiles](#) | Location of compressed source code | `s3://<bucket-name>/builds/<compressed directory name>` |
| <a name="spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON"></a>[spark.emr-serverless.driverEnv.PYSPARK_DRIVER_PYTHON](#) | Necessary for Spark to access your environment | `./environment/bin/python` |
| <a name="spark.emr-serverless.driverEnv.PYSPARK_PYTHON"></a>[spark.emr-serverless.driverEnv.PYSPARK_PYTHON](#) | Necessary for Spark to access your environment | `./environment/bin/python` |
| <a name="spark.executorEnv.PYSPARK_PYTHON"></a>[spark.executorEnv.PYSPARK_PYTHON](#) | Necessary for Spark to access your environment. | `./environment/bin/python` |

For the CLI option, you can use the `aws emr-serverless start-job-run` command with the parameters above configured like [here](https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/jobs-cli.html). This will then look something like this (I omitted some parameters for clarity):

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

The `sparkSubmitParameters` are just an example. The most important bits are the entryPoint and the entryPointArguments, so the file you want to run on the cluster and the arguments you want to pass to it.

You can configure an awful lot when submitting a job. One of the options that I find quite useful is determining a location in your S3 bucket where the log files of the jobs are stored; they help immensely in case of troubleshooting. 

To do that, you can tick the corresponding box in the job submissions form in the management console, or provide the following `--configuration-overrides` besides the other configurations above: 

~~~
{
    "monitoringConfiguration": {
        "s3MonitoringConfiguration": {
            "logUri": "s3://<your-s3bucket>/logs/"
        }
    }
}
~~~

It will create a log folder which then nicely sorts the logs, first based on application id, then job id and the Spark Driver/Executor, so that you don't drown in a sea full of log files (not a nice feeling, I can assure you).
## Monitoring your running jobs

You can use the console to monitor your applications and jobs. The only time I use this frequently is to see how a job is going without leaving the terminal, but you can do a lot more with it in case you want to.

~~~
aws emr-serverless get-job-run \
    --application-id <application-id> \
    --job-run-id <job-run-id>
~~~

You get the `job-run-id` as output in your terminal if you submit a job via the CLI or you can see it in the management console under the details of your job.


## Closing thoughts

I hope that this post has given you the tools to use our Terraform module and start your journey with EMR Serverless. We are always interested in hearing feedback about the module or potential additions you would find helpful, so if you would like to see some particular functionality either drop me a mail or open an [issue on GitHub](https://github.com/kierandidi/terraform-aws-emrserverless/issues). Thanks for reading!