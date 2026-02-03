# Create the bucket

aws s3api create-bucket --bucket dbof --endpoint https://s3-west.nrp-nautilus.io

# Policy

## Generate policy

JXP copied an existing example from the ulmo Repo in nautilus/

cp ../../../ulmo/nautilus/s3_viirs_policy.json s3_dbof_policy.json

### Edit Resource to point to dbof instead of viirs

## Add policy

aws --endpoint-url https://s3-west.nrp-nautilus.io s3api put-bucket-policy --bucket dbof --policy file://s3_dbof_policy.json

## Check policy

aws --endpoint https://s3-west.nrp-nautilus.io s3api get-bucket-policy --bucket dbof --query Policy --output text

# Users

## Check access

aws s3api  get-bucket-acl --bucket dbof --endpoint https://s3-west.nrp-nautilus.io 

### Initially, I see:

{
    "Owner": {
        "DisplayName": "Xavier Prochaska",
        "ID": "profx"
    },
    "Grants": [
        {
            "Grantee": {
                "DisplayName": "Xavier Prochaska",
                "ID": "profx",
                "Type": "CanonicalUser"
            },
            "Permission": "FULL_CONTROL"
        }
    ]
}

### That shows I have full access

## Add additional users

### This is the command.  You append new users with id=USER_ID to the list

aws s3api put-bucket-acl --profile default --bucket dbof --grant-full-control id=profx,id=http://cilogon.org/serverE/users/388729,id=http://cilogon.org/serverE/users/448223,id=http://cilogon.org/serverE/users/340867 --endpoint https://s3-west.nrp-nautilus.io

## Check access

aws s3api  get-bucket-acl --bucket dbof --endpoint https://s3-west.nrp-nautilus.io 

