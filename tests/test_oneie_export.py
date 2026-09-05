import unittest
from src.oneie.export import export_record
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.event import Event
from src.unified_format.trigger import Trigger
from src.unified_format.argument import Argument


class Tokenizer:
    unk_token='[UNK]'
    cls_token_id=101
    sep_token_id=102
    def tokenize(self,t): return [t]
    def convert_tokens_to_ids(self,t):return list(range(len(t)))
    def encode(self,t,add_special_tokens=True):return [101]+self.convert_tokens_to_ids(t)+[102]


class ExportTests(unittest.TestCase):
    def test_links_and_subword_alignment(self):
        sample=EventExtractionData('x','A B',tokens=['A','B'],events=[
            Event('move',[Trigger('A',(0,1))],[Argument('place',[{'text':'B','span':(1,2)}])])])
        result=export_record(sample,{},Tokenizer(),'x')
        self.assertEqual(result['token_lens'],[1,1])
        self.assertEqual(result['event_mentions'][0]['arguments'][0]['entity_id'],result['entity_mentions'][0]['id'])
        self.assertEqual(result['entity_mentions'][0]['end'],2)

    def test_overlaps_are_audited_and_links_are_not_dangling(self):
        sample=EventExtractionData('x','A B',tokens=['A','B'],events=[
            Event('e',[Trigger('A',(0,1))],[Argument('r',[{'text':'A B','span':(0,2)},{'text':'B','span':(1,2)}])])])
        result=export_record(sample,{},Tokenizer(),'x')
        self.assertEqual(len(result['entity_mentions']),1)
        self.assertEqual(len(result['event_mentions'][0]['arguments']),1)
        self.assertEqual(len(result['conversion_notes']),2)

    def test_negative_sentence_is_retained(self):
        result=export_record(EventExtractionData('x','A',tokens=['A']),{},Tokenizer(),'x')
        self.assertEqual(result['event_mentions'],[])

if __name__=='__main__':unittest.main()
