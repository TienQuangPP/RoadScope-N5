# Security Policy

## Supported Version

Security fixes target the current default branch and released ONNX artifact.

## Reporting a Vulnerability

Use GitHub private vulnerability reporting when it is enabled for the
repository. Do not place credentials, private datasets, personal images, or a
working exploit in a public issue.

Include the affected file, environment, reproduction steps, impact, and a
minimal non-sensitive example. Public disclosure should wait until a fix or
mitigation is available.

## Security Boundaries

The CLI reads local images and writes local results. It does not require a
network connection or API credential. Treat ONNX files and image inputs from
untrusted sources as untrusted data and process them in an isolated
environment.

PyTorch checkpoint files use a serialization format that can execute code
when loaded. Only pass trusted `.pt`, `.pth`, or `.ckpt` files to training,
evaluation, export, or benchmark commands. Prefer the released ONNX artifact
for inference-only use.

Model outputs are not security, safety, identity, or authorization decisions.
