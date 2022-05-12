# masKIT: Ensemble-based gate dropouts for quantum circuits
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

MasKIT is a framework that provides masking functionality in the context of
parameterized quantum circuits (PQC) for PennyLane. It targets *scientists* and
simplifies researching trainability and expressivity of circuits by enabling to
dynamically mask gates within the circuit. The framework is designed to act as
a drop-in replacement and therefore allows to enhance your existing PennyLane
projects with low effort.

The masking is supported on different axes, i.e. layers, wires, parameters, and
entangling gates, for different modes, i.e. adding, removing, inverting.

  The current version is still in a development stage and therefore does not cover
  the whole functionality one might imagine for masking PQCs.
  Please feel invited to submit your contributions and ideas.

## Installation

The framework can be installed via pypi:

```bash
python -m pip install maskit
```

## Contributing

You love research as much as we do? Anything missing? We welcome all support,
whether on bug reports, feature requests, code, reviews, tests, documentation,
blog posts, and more.
Please have a look at our [contribution guidelines](docs/CONTRIBUTING.md).

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/cDenius"><img src="https://avatars.githubusercontent.com/u/28619054?v=4?s=100" width="100px;" alt=""/><br /><sub><b>cDenius</b></sub></a><br /><a href="https://github.com/cirKITers/masKIT/commits?author=cDenius" title="Code">💻</a> <a href="#ideas-cDenius" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-cDenius" title="Maintenance">🚧</a> <a href="https://github.com/cirKITers/masKIT/issues?q=author%3AcDenius" title="Bug reports">🐛</a> <a href="https://github.com/cirKITers/masKIT/pulls?q=is%3Apr+reviewed-by%3AcDenius" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/eileen-kuehn"><img src="https://avatars.githubusercontent.com/u/8090701?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Eileen Kuehn</b></sub></a><br /><a href="https://github.com/cirKITers/masKIT/commits?author=eileen-kuehn" title="Code">💻</a> <a href="#ideas-eileen-kuehn" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-eileen-kuehn" title="Maintenance">🚧</a> <a href="https://github.com/cirKITers/masKIT/commits?author=eileen-kuehn" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/maxfischer2781"><img src="https://avatars.githubusercontent.com/u/5708444?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Max Fischer</b></sub></a><br /><a href="https://github.com/cirKITers/masKIT/pulls?q=is%3Apr+reviewed-by%3Amaxfischer2781" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/nikmetz"><img src="https://avatars.githubusercontent.com/u/23529838?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Niklas Metz</b></sub></a><br /><a href="https://github.com/cirKITers/masKIT/commits?author=nikmetz" title="Code">💻</a> <a href="https://github.com/cirKITers/masKIT/commits?author=nikmetz" title="Tests">⚠️</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
