namespace PaddleOcrSharp.Cli;

/// <summary>Usage output.</summary>
public static class Help
{
    /// <summary>Prints the usage banner.</summary>
    public static int Print()
    {
        Console.WriteLine(
            """
            paddleocr-sharp — pure C# PaddleOCR-VL

            Usage:
              paddleocr-sharp download [model ...]        Fetch models into the local cache
              paddleocr-sharp parse <image|pdf ...>       Parse pages to markdown / JSON
              paddleocr-sharp recognize <image>           Recognise one already-cropped block
              paddleocr-sharp bench                       Measure throughput of the model stages

            Common options:
              --model-dir <path>      Use a checkpoint directory instead of the cache
              --cache <path>          Override the cache root (default: ~/.cache/paddleocr-sharp)
              --endpoint <url>        Hugging Face-compatible endpoint (default: HF_ENDPOINT)

            parse options:
              --layout-dir <path>     Use a PP-DocLayoutV3 directory instead of the cache
              --no-layout             Send the whole page to the VL model
              --layout-threshold <f>  Detection score threshold (default: 0.3)
              --chart                 Recognise charts instead of keeping them as images
              --seal                  Recognise seals instead of keeping them as images
              --ocr-images            Run OCR over image blocks too
              --doc-orientation       Rotate pages upright before parsing
              --doc-unwarping         Flatten curled pages before parsing
              --prompt-label <name>   Whole-page mode label when --no-layout (e.g. spotting)
              --dpi <n>               PDF rendering resolution (default: 200)
              --max-pages <n>         Stop after this many PDF pages
              --password <text>       Password for an encrypted PDF
              --output-dir <path>     Write <name>.md, <name>.json and imgs/ per page
              --format markdown|json  Output format when writing to stdout
              --page-separator <text> Text between pages (default: a blank line)
              --block-concurrency <n> Blocks recognised in parallel (default: 1)

            recognize options:
              --prompt-label <name>   ocr | table | formula | chart | seal | spotting  (default: ocr)
              --prompt <text>         Raw instruction, overrides --prompt-label
              --max-new-tokens <n>    Token budget (default: 8192)
              --temperature <f>       0 selects greedy decoding (default: 0)
              --top-p <f>             Nucleus mass; ignored when greedy
              --repetition-penalty <f>
              --min-pixels <n>        Lower pixel budget (default: 112896)
              --max-pixels <n>        Upper pixel budget (default: 1003520)
              --output <path>         Write the result to a file instead of stdout

            bench options:
              --width <n> --height <n>   Synthetic page size (default: 1024x1024)
              --iterations <n>           Repeats per stage (default: 3)
              --no-vl                    Skip the vision tower and decoder
              --no-layout                Skip the layout graph
            """);

        return 0;
    }

    /// <summary>Reports an unknown verb.</summary>
    public static int Unknown(string verb)
    {
        Console.Error.WriteLine($"Unknown command '{verb}'.");
        Print();
        return 2;
    }
}
