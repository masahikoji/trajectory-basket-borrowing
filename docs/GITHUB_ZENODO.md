# GitHub and Zenodo release notes

This repository is structured so that the same tagged GitHub release can be archived by Zenodo.

Recommended release sequence:

1. Push the repository to GitHub.
2. Confirm `CITATION.cff`, the author list, and the release tag (for example `v1.0.0`).
3. Enable the repository in the Zenodo GitHub integration.
4. Create the GitHub release from the verified commit/tag.
5. Confirm the Zenodo record metadata and version-specific DOI after archiving.

`CITATION.cff` is intentionally kept at the repository root so GitHub can display citation information and Zenodo can read software metadata. A `.zenodo.json` file is not included because no Zenodo-specific community, grant, or related-identifier metadata are required by the code package itself.

No license file is included in this source archive. Select and add the intended software license before public release if reuse permissions should be granted explicitly.

Zenodo documentation:

- https://help.zenodo.org/docs/github/
- https://help.zenodo.org/docs/github/describe-software/citation-file/
- https://help.zenodo.org/docs/github/archive-software/github-upload/
