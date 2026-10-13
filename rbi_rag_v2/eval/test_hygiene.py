import unittest
from generation.dspy_advisor import sanitize_output

class TestOutputHygiene(unittest.TestCase):
    def test_sanitize_dspy_markers(self):
        dirty_input = "Under Basel III, CRAR is 8% [[ ## completed ## ]]. [Block ID: 1] Block ID: 12"
        clean_output = sanitize_output(dirty_input)
        
        self.assertNotIn("[[ ## completed ## ]]", clean_output)
        self.assertNotIn("Block ID: 1", clean_output)
        self.assertNotIn("Block ID: 12", clean_output)
        self.assertEqual(clean_output, "Under Basel III, CRAR is 8%.")

    def test_sanitize_thinking_tags(self):
        dirty_input = "<Thinking>Analyzing capital adequacy requirements...</Thinking>The requirement is 8%."
        clean_output = sanitize_output(dirty_input)
        
        self.assertNotIn("<Thinking>", clean_output)
        self.assertNotIn("</Thinking>", clean_output)
        self.assertEqual(clean_output, "The requirement is 8%.")

if __name__ == "__main__":
    unittest.main()
