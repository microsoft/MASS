#from fairseq.data import BertDictionary

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from .bert_dictionary import BertDictionary


@register_task('translation_mass')
class TranslationMASSTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return min(self.args.max_source_positions, self.args.max_target_positions)
