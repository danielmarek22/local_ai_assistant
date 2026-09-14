# Vendored browser dependencies

ASTRA serves these browser modules locally so the UI can start without internet
access. The directory names pin the versions that were previously loaded from
jsDelivr:

- Three.js 0.169.0
- `@pixiv/three-vrm` 3.0.0
- Marked 13.0.2
- DOMPurify 3.1.6

Only runtime files reachable from ASTRA's browser entry point are included. Each
package's upstream license is stored alongside its files. Update a dependency by
copying files from the corresponding npm package, preserving its license, and running
the complete JavaScript test suite. `test_frontend_dependencies.mjs` verifies that the
browser import graph is local and complete.
