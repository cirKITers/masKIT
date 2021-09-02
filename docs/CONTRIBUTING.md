<!-- 
- Steps for creating good issues or pull requests.
- Links to external documentation, mailing lists, or a code of conduct.
- Community and behavioral expectations. 
-->

# Contributing to masKIT

First off, thank you for considering contribution to masKIT.
It's people like you that make great research!

The following is a set of guidelines for contributing to masKIT, which is hosted in the [cirKITers Organisation](https://github.com/cirkiters) on [GitHub](https://github.com/cirKITers/masKIT). 
These are mostly guidelines, not rules. 
Use your best judgment, and feel free to propose changes to this document in a pull request.

## How to report a bug

So you have encountered a bug – that’s bad – and you have decided to report it – that’s great! 
Now, before reporting the bug take a moment to look through [previously reported bugs](https://github.com/cirkiters/maskit/issues?q=label%3Abug) if it was reported already.

- Does a report exist that provides a solution? Great, go right ahead using it and enjoy masKIT.
- Does a report exist but is still open? 
  If it lacks some information that you can provide, leave a comment. 
  Reports that seem not to be worked on might also benefit from a quick comment to raise awareness. 
  Either way, consider subscribing to the report to be notified on progress.
- Does no report exist on the bug? 
  Open one and provide the necessary information to look into your problem.
  - What version of masKIT and Python are you using? 
    Have you tried another version, and did the bug occur there as well?
  - How can the issue be reproduced reliably? 
    Try to strip away all parts that are not needed to reproduce the issue – the more minimal and reproducible your example, the easier it is to help you.
  - What did you expect to see and what did you see instead?
    
If you find a solution to the bug while coming up with an example, please open a report anyway.

## How to suggest a feature or enhancement

If you find yourself wishing for a feature that doesn't exist in masKIT, you are probably not alone.
New features will provide a range of opportunities for new research, so please let us know.
Before reporting your proposal take a moment to look through the [previously suggested features](https://github.com/cirkiters/maskit/issues?q=label%3Aenhancement) if it has already been reported.

- Does a report exist but is still open? If it lacks information that you can provide, leave a comment. 
  Reports that seem not to be worked on might also benefit from a quick comment to raise awareness.
  Either way, consider to subscribe to the report to be notified on progress.
- Does no report exist on the feature? 
  Open an issue on our issues list on GitHub which describes the feature you would like to see, why you need it, and how it should work.
  - In case there are any publications on the feature, please provide links to webpages, papers or similar resources that put your suggestion into context.
  - Can you describe the feature as an API, algorithm or code? The better you can sketch out the feature, the easier it is to discuss and implement.

## Contributing Fixes and Features

So you want to actively contribute to masKIT – You. Are. Awesome! 🥳 Now, with every great contribution comes great responsibility, so here are some steps to help you leave masKIT a little better than you found it.

- In case you just want to get started contributing to open source research projects, check out the [recommended first issues](https://github.com/cirkiters/maskit/contribute). 
  These features are similar to existing features of masKIT or build on common programming principles – perfect to learn the ropes of contributing without worrying much about technical details.
- Every contribution should have an associated issue – check out how to [suggest enhancements](#how-to-suggest-a-feature-or-enhancement) and [report bugs](#how-to-report-a-bug) to find and possibly open an issue as appropriate.
  This allows you to get feedback before investing too much time, and shows to others that you are tackling the issue.

To actually contribute to the masKIT repository, you need to open and maintain a Pull Request.
By sticking to the masKIT quality criteria and responding to feedback on the PR.

### Managing a Pull Request

A Pull Request allows you to commit your changes to a copy of the masKIT repository, and request them to be merged ("pulled") into the masKIT repository.
If you are not familiar with creating Pull Requests, check out the guides on [forks](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-forks) and [pull requests from forks](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork).

When opening a Pull Request, make sure to provide the information needed to understand your proposal.

- Use a title that summarises the contribution.
  By convention, use imperative mood such as "Add new dataset".
- In the Pull Request description, give an outline of your contribution.
  - When the contribution consists of distinct elements, add a task list.
    You can check off tasks as they are completed, allowing you and us to track progress.
  - Refer to issues and other Pull Requests affected by your contribution.
    Make sure to mark the corresponding ticket of your contribution, such as "Closes #9." (assuming your issue is #9).

After opening the Pull Request, respond to feedback and make new commits as needed.
Once you and us are happy with the contribution, it will be [squash merged](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-merges#squash-and-merge-your-pull-request-commits); so don't sweat it – we can just rewrite history to fix any errors made along the way!

### Keeping Quality high

Having new things is great, but they must also fit to all the rest.
There are some formal and informal quality criteria we ask you to stick to for every contribution:

- Code must be formatted to conform to the black and flake8 tools.
  This is enforced by the repository.
  You can locally check your code by running the tools yourself or using the preconfigured precommit hooks.
  In most cases, black is capable of reformatting code adequately.

```bash
# use black to reformat code
python3 -m black maskit tests
# check for remaining code smells
python3 -m flake8 maskit tests
```

- Code must pass all existing unittests via pytest for Python 3.7 and upwards.
  This is enforced by the repository.
  You can locally check your code by running the tools yourself.

```bash
# evaluate unit tests
python3 -m pytest
```

- Code should be covered by unittests.
  This is checked but not enforced by the repository. 
  If you contribution is similar to an existing feature, take the latter's unittests as a template; if not, we will discuss with you how to best approach unittests.
- Any user-facing feature should be documented.
  The documentation is compiled using sphinx from the ./docs directory.
  If your contribution is similar to an existing feature, take the latter's documentation as a template; if not, we will discuss with you how to best approach documentation.

Phew! That was a lot to read! Now go out there and put that knowledge to good use – we are happy to help you along the way.
