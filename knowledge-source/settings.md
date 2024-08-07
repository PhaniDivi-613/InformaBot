---

copyright:
  years: 2019, 2024
lastupdated: "2024-05-07"

keywords:

subcollection: atracker

---

{{site.data.keyword.attribute-definition-list}}


# Configuring account settings
{: #settings}

You can configure the {{site.data.keyword.atracker_short}} account settings in your account by using the {{site.data.keyword.atracker_short}} CLI, the {{site.data.keyword.atracker_short}} REST API, and Terraform scripts. Set these settings to define where and how auditing events are collected, routed, and managed in your account.
{: shortdesc}

When you configure or modify the {{site.data.keyword.atracker_short}} account settings, consider the following information:

- Every time you modify the {{site.data.keyword.atracker_short}} account settings, the data that is passed in the new request replaces any existing configuration data. You must ensure that any existing data is not deleted when you run an update of the account settings by including it in the new request.
{: important}

- Before you disable public endpoints by setting `--private-api-endpoint-only TRUE`, make sure your account has access to the private endpoint.  You can do this by running the command `ibmcloud account show`.  If `VRF Enabled` is `true` and `Service Endpoint Enabled` is `true` then you have access to the private endpoint.  If you do not have access to the private endpoint, you will be unable to re-enable the public endpoint since private endpoint access is required to re-enable the public endpoint.
{: important}



## What data can you configure through the {{site.data.keyword.atracker_full}} account settings?
{: #settings-what}

You can define any of the following information:
1. The location in your {{site.data.keyword.cloud_notm}} account where the {{site.data.keyword.atracker_short}} account configuration metadata is stored.

    By metadata, we refer to the target/route/settings data that is available across the account in any region.

    You can choose any of the supported locations where {{site.data.keyword.atracker_short}} is available. For more information, see [Locations](/docs/atracker?topic=atracker-regions&interface=cli).

    Take into account any corporate or industry compliance requirements such as Financial Services Validated locations, or EU-managed regions.

2. The type of endpoints that are allowed to manage the {{site.data.keyword.atracker_short}} account configuration in the account.

    You can configiure public endpoints, private endpoints, or both.

3. The locations where an account administrator can define targets to collect auditing events.

    You can choose any of the supported locations where {{site.data.keyword.atracker_short}} is available. For more information, see [Locations](/docs/atracker?topic=atracker-regions&interface=cli).

    Take into account any corporate or industry compliance requirements such as Financial Services Validated locations, or EU-managed regions.

4. 1 or more targets in the account that will collect auditing events from supported {{site.data.keyword.atracker_short}} locations where you have not configured how you want to collect the auditing data.

    If you define more than 1 target, all default targets get a copy of auditing events that do not have a routing rule to indicate where to collect them in the account. You can define up to 2 default targets per account.
    {: note}


## IAM permissions
{: #settings_access}

You must grant users IAM permissions to manage the account settings. For more information, see [Assign access to resources](/docs/account?topic=account-assign-access-resources).

When you define a policy, you can must set the scope of the policy to the account. A route is a global resource that is not bound to a specific region.
{: note}

| IAM action               | IAM Policy scope  | IAM Roles                          | Description         |
| ------------------------ | ----------------- | ---------------------------------- | -------------- |
| `atracker.setting.get`    | Account        | `Administrator`  \n `Editor`  \n `Viewer`  \n `Operator` | Get settings information |
| `atracker.setting.update` | Account        | `Administrator`| Update settings |
{: caption="Table 1. Required IAM roles}




## CLI prerequisites
{: #settings-prereqs-cli}
{: cli}

Before you use the the CLI to manage {{site.data.keyword.atracker_short}} account settings, [Install the {{site.data.keyword.atracker_full_notm}} CLI](/docs/atracker?topic=atracker-atracker-cli-config).

Check that you have IAM permissions to read or update the {{site.data.keyword.atracker_short}} account settings.


## Getting account settings using the CLI
{: #settings-get-cli}
{: cli}

Use this command to get the settings for the {{site.data.keyword.atracker_full_notm}} account configurations.

```sh
ibmcloud atracker setting get [--output FORMAT]
```
{: pre}

### Command options
{: #settings-get-options}

`--output FORMAT`
:   If `JSON` is specified, output will be returned in JSON format.  If `JSON` is not specified, output will be returned in a tabular format.

`help` | `--help` | `-h`
:   List options available for the command.


### Example
{: #settings-get-example}

The following is an example where no default or permitted targets have been set and the API version is V2.

```text
Atracker settings
Metadata region primary:     us-south
Metadata region backup:      us-east
Default targets:             []
Permitted target regions:    []
Private api endpoint only:   false
API version:                 2
```
{: codeblock}

## Updating settings using the CLI
{: #settings-update-cli}
{: cli}

Use this command to modify current account settings such as the default targets, permitted target regions, and the primary metadata region.

```sh
ibmcloud atracker setting update [--metadata-region-primary REGION] [--metadata-region-backup REGION] [--default-targets TARGET] [--permitted-target-regions REGIONS] [--private-api-endpoint-only ( TRUE | FALSE )] [--output FORMAT] [--force]
```
{: pre}

### Command options
{: #settings-update-options}

`default-targets`
:   Is a list of target IDs.  If no routing rules cause events to be sent to other targets, these targets will received the events.  TARGETS is a comma-separated list of target IDs.

`permitted-target-regions`
:   Is the list of regions that can be used to define a target. REGIONS is a comma-separated list of regions. A maximum of two permitted target regions can be specified.

`metadata-region-primary`
:   Specify the REGION where the metadata associated with route and target definitions is stored.

`metadata_region_backup`
:   Is the region where the metadata associated with route and target definitions is stored as a backup location.

`private-api-endpoint-only`
:   Specifies whether nor not a private endpoint can be used.  If `true` only a private endpoint can be used.

`--output FORMAT`
:   If `JSON` specified, output will be returned in JSON format.  If `JSON` is not specified, output will be returned in a tabular format.

 `help` | `--help` | `-h`
:   List options available for the command.

If the update is successful, the current settings will be displayed.


## API settings and actions
{: #settings-api}
{: api}

The following table lists the actions that you can run to manage settings:

| Action                     | REST API Method  | API_URL                                          |
|----------------------------|------------------|--------------------------------------------------|
| Get settings information            | `GET`           | `<ENDPOINT>/api/v2/settings`              |
| Update settings            | `PUT`            | `<ENDPOINT>/api/v2/settings`  |
{: caption="Table 1. Settings actions by using the {{site.data.keyword.atracker_full_notm}} REST API" caption-side="top"}

You can use private and public endpoints to manage settings. For more information about the list of `ENDPOINTS` that are available, see [Endpoints](/docs/atracker?topic=atracker-endpoints).

* By default, you can manage settings from the private network. You must use an API endpoint with the following format: `https://private.<region>.atracker.cloud.ibm.com`

* You can also enable public endpoints in a region to manage settings. For more information, see [Managing endpoints](/docs/atracker?topic=atracker-endpoints_manage_v2).

For more information about the REST API, see [the settings API](/apidocs/atracker#get-settings){: external}.
{: note}

## API prerequsites
{: #settings-prereqs-api}
{: api}

To make API calls to manage settings, complete the following steps:
1. Get an IAM access token. For more information, see [Retrieving IAM access tokens](/docs/atracker?topic=atracker-retrieve-iam-token).
2. Identify the API endpoint in the region where you plan to configure or manage settings. For more information, see [Endpoints](/docs/atracker?topic=atracker-endpoints).


## Getting settings using the API
{: #settings-get-api}
{: api}

You can use the following cURL command to get existing settings information:

```shell
curl -X GET  ENDPOINT/api/v2/settings   -H "Authorization:  $ACCESS_TOKEN"
```
{: codeblock}

Where:

`ENDPOINT`
:   Is the API endpoint in the region where you plan to configure or manage a target. For more information, see [Endpoints](/docs/atracker?topic=atracker-endpoints).

For example, you can use the following cURL request settings information the US South region:

```shell
curl -X GET   https://private.us-south.atracker.cloud.ibm.com/api/v2/settings   -H "Authorization:  $ACCESS_TOKEN"
```
{: screen}

A response similar to the following is returned:

```json
{
 "default_targets": ["50375218-0000-4234-bbb4-171bebab8408", "c7519d8a-5f97-498b-0000-8542f60955cd"],
 "permitted_target_regions": ["us-south", "us-east"],
 "metadata_region_primary": "us-south",
 "metadata_region_backup": "eu-de",
 "private_api_endpoint_only": false
}
```
{: screen}

Where:

`default_targets`
:   Is a list of target IDs.  If no routing rules cause events to be sent to other targets, these targets will received the events.

`permitted_target_regions`
:   Is the list of regions that can be used to define a target. A maximum of two permitted target regions are allowed.

`metadata_region_primary`
:   Is the region where the metadata associated with route and target definitions is stored.

`metadata_region_backup`
:   Is the region where the metadata associated with route and target definitions is stored as a backup location.

`private_api_endpoint_only`
:   Specifies whether nor not a private endpoint can be used.  If `true` only a private endpoint can be used.


## Updating settings using the API
{: #settings-update-api}
{: api}

When you update settings, you must include the settings information in the data section of the request.
- You must pass all fields.
- Update the fields that need changing.

You can use the following cURL command to update settings:

```shell
curl -X PUT  <ENDPOINT>/api/v2/settings
-H "Authorization:  $ACCESS_TOKEN"
-H "content-type: application/json"
-d '{
   "default_targets": ["IDs"],
   "permitted_target_regions": ["REGIONS"],
   "metadata_region_primary": "REGION",
   "metadata_region_backup": "REGION",
   "private_api_endpoint_only": false
}'
```
{: codeblock}

Where:

`ENDPOINT`
:    Is the API endpoint in the region where you plan to configure or manage a target. For more information, see [Endpoints](/docs/atracker?topic=atracker-endpoints).

`default_targets`
:   Is a list of target IDs.  If no routing rules cause events to be sent to other targets, these targets will received the events.

`permitted_target_regions`
:   Is the list of regions that can be used to define a target. A maximum of two permitted target regions can be specified.

`metadata_region_primary`
:   Is the region where the metadata associated with route and target definitions is stored.

`metadata_region_backup`
:   Is the region where the metadata associated with route and target definitions is stored as a backup location.

`private_api_endpoint_only`
:   Specifies whether nor not a private endpoint can be used.  If `true` only a private endpoint can be used.



## HTTP response codes
{: #settings-rc}
{: api}

When you use the {{site.data.keyword.atracker_full_notm}} REST API, you can get standard HTTP response codes to indicate whether a method completed successfully.

- A 200 response always indicates success.
- A 4xx response indicates a failure.
- A 5xx response usually indicates an internal system error.

See the following table for some HTTP response codes:

| Status code | Status | Description |
|-------------|--------|-------------|
| `200` |	OK | The request was successful. |
| `201` |	OK | The request was successful. A resource is created. |
| `400` | Bad Request |	The request was unsuccessful. You might be missing a parameter that is required. |
| `401` |	Unauthorized | The IAM token that is used in the API request is invalid or expired. |
| `403` |	Forbidden | The operation is forbidden due to insufficient permissions. |
| `404` | Not Found |	The requested resource doesn't exist or is already deleted. |
| `429` |	Too Many Requests |	Too many requests hit the API too quickly. |
| `500` |	Internal Server Error |	Something went wrong in {{site.data.keyword.atracker_full_notm}} processing. |
{: caption="Table 2. List of HTTP response codes" caption-side="top"}

## Getting account settings using the UI
{: #settings-get-ui}
{: ui}

1. [Log in to your {{site.data.keyword.cloud_notm}} account](https://cloud.ibm.com/login){: external}.
2. Click the **Menu** icon ![Menu icon](../icons/icon_hamburger.svg) &gt; **Observability**.
3. Select **Activity Tracker**.
4. Select **Routing**.
5. Select **Settings**.

On this page you can view the following settings:
- **Metadata location**: Displays Primary metadata region and Backup metadata region.
- **Permitted target regions**. Displays the targets regions where events can be sent.
- **Default targets**: Displays the configured default targets.
- **Public endpoints**: Displays if public endpoints are enabled. When disabled, the {{site.data.keyword.atracker_short}} UI cannot be accessed.
- **Reports**: Displays the configuration in JSON format.

## Updating settings using the UI
{: #settings-update-ui}
{: ui}

1. [Log in to your {{site.data.keyword.cloud_notm}} account](https://cloud.ibm.com/login){: external}.
2. Click the **Menu** icon ![Menu icon](../icons/icon_hamburger.svg) &gt; **Observability**.
3. Select **Activity Tracker**.
4. Select **Routing**.
5. Select **Settings**.

Click **Edit** next to the setting to be changed. You can modify the following settings:
- **Metadata location**: Select your desired Primary metadata region and Backup metadata region.
- **Permitted target regions**: Select the region where targets can be created. If no regions are selected, then all regions can receive events.
- **Default targets**: Select the target that will be used by default when routing rules do not exist or are not matched.
