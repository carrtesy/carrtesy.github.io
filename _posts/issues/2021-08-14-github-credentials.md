---
title: "[Issues] Token Authentication for Github"

date: 2021-08-14
last_modified_at: 2021-08-14
categories: 
 - github
tags:
 - github
 - token authentication
use_math: true
---



Issues related to failure of password authentication on github have occurred.

You may have seen something like:

```
Password authentication is temporarily disabled as part of a brownout. 

Please use a personal access token instead.
```

This post would tell the way to deal with those issues.



## Why It Happens

To paraphrase, github no longer accepts password as Git authentication mechanism. Rather, we should use token-based authentification. For details, take a look at [here](https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/).



## What should we do

We should instead get tokens from Github.

1. Log in and go to settings.

   ![settings](../../assets/images/issue/0814_github_auth_token/settings)

2. Go to Developer settings.

   ![devset](../../assets/images/issue/0814_github_auth_token/devset)

3. Go to Personal Access Token, and push "generate new token".

   ![token](../../assets/images/issue/0814_github_auth_token/token)

4. Set the detail. "repo" option needs to be checked, as it deals with access for the repository. 

   ![tokendetail](../../assets/images/issue/0814_github_auth_token/tokendetail)

Access key vanishes, so **copy it elsewhere!**

With issued token, we should log in with it instead. 

![mytoken](../../assets/images/issue/0814_github_auth_token/mytoken)

```bash
$ git push origin master
Username for "https://github.com": dongminkim0220
Password for "dongminkim0220": 
```



## If git credential is involved...

If you have automatized the push/pull process using github credentials somehow, you may need to reset the credentials.

Unset by:

```bash
$ git config --global --unset credential.helper
```

Reset by:

```bash
$ git config --global credential.helper cache
```

Log in:

```bash
$ git push origin master
Username for "https://github.com": dongminkim0220
Password for "dongminkim0220": 
```

