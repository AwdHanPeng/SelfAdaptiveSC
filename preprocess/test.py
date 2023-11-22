# pick func from class
from tree_sitter import Language, Parser

language = 'java'

Language.build_library(
    './build/{}.so'.format(language),
    [
        './vendor/tree-sitter-{}'.format(language),
    ]
)
language = Language('./build/{}.so'.format(language), language)
lang_parser = Parser()
lang_parser.set_language(language)
code = "public class A36046937 {\n     private String convertIfStatement(IfWrapper iw, ScriptingContext context) {\n        String condition = iw.getConditionExpression();\n        condition = evaluateNumericExpression(iw.getConditionExpression(), Boolean.class).toString();\n        String s = \"if (\" + condition + \") {\";\n        return s;\n    }\n\n    /**\n     * Convert an 'else' or 'else if' statement to Java.\n     * @param iw document wrapper of the 'if' structure.\n     * @param context validation context.\n     * @return corresponding Java code.\n     */\n    private String convertElseStatement(IfWrapper iw, ScriptingContext context) {\n        String condition = iw.getConditionExpression();\n        if (condition != null) {\n            return \"} else \" + convertIfStatement(iw, context);\n        } else {\n            return \"} else {\";\n        }\n    }\n  \n}"
tree = lang_parser.parse(bytes(code, "utf-8"))

root = None


def get_func_node(node):
    if node.type == 'method_declaration':
        global root
        root = node
    else:
        for child in node.children:
            get_func_node(child)


get_func_node(tree.walk().node)

print(root.text)
b = b"example"
s3 = b.decode()
print(s3)
