## Trashnet

### Project Structure Explanation

- **models**: save all models which need LFS for the models
- **notebooks**: model training experimentation
- **scripts**: script that runs certain functionality (ex: run training)
- **src**: all of the project functions
- **test**: Unit test and itegration test

### Workflow Explanation

**Disclaimer**: To do this CI/CD we need `github lfs`, and for the time being my github lfs bandwith is full. So i will do the demo with local testing github action with `nektos/act`.

Workflow will be split in 3 workflows, `ci.yml`, `cd-tag-ver.yml`, and  `cd-latest-ver.yml`

- **ci.yml - dev branch**:
    - usage:
        - unit test: test all functions used in the script
        - integration test: Making sure model that has been pushed can be used for inference
    - trigger: `pull request dev`, `push dev`

- **cd-tag-ver.yml - tagged version branch**:
    - usage:
        - push tagged version: push tagged version to huggingface, example:
        ```sh
        git tag v1.1
        git push origin v1.1
        ```

        This will push the model and it's config to huggingface.
        | ![cd-tag-ver-1](./documentation/workflow/cd-tag-ver-1.png) | ![cd-tag-ver-2](./documentation/workflow/cd-tag-ver-2.png) |
        |-----------------------------------------------------------|-----------------------------------------------------------|
    - trigger: `tag v1.x` -> `push v1.x`
    
- **cd-latest-ver.yml - main branch**:
    - usage:
        - push latest model: push latest model to the root directory of huggingface repository.
    - trigger: `pull request main`, `push main`
        ![cd-latest-ver](./documentation/workflow/cd-latest-ver.png)
    

