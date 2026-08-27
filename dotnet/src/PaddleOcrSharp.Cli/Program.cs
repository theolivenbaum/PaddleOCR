using PaddleOcrSharp.Text;

if (args.Length >= 2 && args[0] == "tokdebug")
{
    var tokenizer = BpeTokenizer.FromFile("/home/user/ref/vl16/tokenizer.json");
    foreach (string text in args[1..])
    {
        List<int> ids = tokenizer.Encode(text);
        Console.WriteLine($"{text} -> [{string.Join(", ", ids)}] {string.Join("|", ids.Select(tokenizer.IdToToken))}");
    }

    Console.WriteLine("O=" + tokenizer.TokenToId("O") + " C=" + tokenizer.TokenToId("C")
        + " R=" + tokenizer.TokenToId("R") + " OC=" + tokenizer.TokenToId("OC")
        + " CR=" + tokenizer.TokenToId("CR") + " OCR=" + tokenizer.TokenToId("OCR"));
    return 0;
}

Console.WriteLine("paddleocr-sharp");
return 0;
