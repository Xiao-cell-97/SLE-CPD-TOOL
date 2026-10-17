# Publish the English tool on GitHub Pages

1. Confirm permission to publish the models and code; complete the copyright attribution and appropriate licensing.
2. Sign in to GitHub and create a public repository, for example sle-cpd-score-tool. Public repositories expose their contents.
3. Use Add file → Upload files to upload the extracted package contents. Keep index.html at the repository root, not inside an extra enclosing folder. Do not upload only the ZIP archive.
4. Commit the files to the main branch.
5. Open Settings → Pages. Under Build and deployment, select Deploy from a branch, choose main and /(root), then Save.
6. Wait for deployment to complete, then use the actual address shown under Settings → Pages → Visit site. Publication may take approximately 10 minutes.
7. Test both models with synthetic measurements on desktop and mobile. Verify required inputs, explicit imputation, clearing, model switching and JSON export.

A typical address has the form https://YOUR-USERNAME.github.io/sle-cpd-score-tool/ . This is an example format, not a deployed URL. Network accessibility varies; the offline HTML does not depend on GitHub availability.

This package runs equivalent tree inference in the browser and needs no Python prediction server. Static publication does not make it a clinically validated calculator. Exclude patient data, credentials, and individual prediction tables from the public repository.

Reference documentation, checked 2026-10-16:
- https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
