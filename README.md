# Code2Comment
Given a code snippet, generate comment automatically.
### Data

Question & Answer pairs on Stackoverflow, python code only (Java code already experimented in [previous work](https://github.com/sriniiyer/codenn))

### Preprocessing
see [another repo](https://github.com/shwinshaker/Code2Comment-SBT)

### Model

Sequence to sequence model with attention

### Example results
* code snippet:  `django.template.loader.get_template(template_name)`
* target comment:  "check if a template exists in Django"
* generated comment:  "get a template in Django"

### See our report above for more
