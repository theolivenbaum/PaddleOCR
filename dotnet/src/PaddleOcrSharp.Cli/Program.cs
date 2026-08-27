using PaddleOcrSharp.Cli;

CommandLine command = CommandLine.Parse(args);

return command.Verb.ToLowerInvariant() switch
{
    "download" => await DownloadCommand.RunAsync(command),
    "recognize" or "recognise" => await RecognizeCommand.RunAsync(command),
    "bench" => await BenchCommand.RunAsync(command),
    "" or "help" or "--help" or "-h" => Help.Print(),
    _ => Help.Unknown(command.Verb),
};
