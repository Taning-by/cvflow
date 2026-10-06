; CVFlow 安装程序（Inno Setup 6）
;
; 一个安装包，GPU 推理支持做成可勾选的组件：
;   主程序      约 500 MB，装完就能用，推理跑在 CPU 上
;   GPU 推理支持 约 1.9 GB，onnxruntime 的 CUDA provider（325 MB）
;                + CUDA 12 / cuDNN 9 的运行库（约 1.6 GB）
;   示例方案    examples 下的示例流程和图片，装到 ProgramData 里
;
; 为什么能这么切：onnxruntime 的 CUDA 后端是单独一个 DLL，运行时才去加载。不装它
; （以及 CUDA/cuDNN 运行库）时，软件照常启动，只是 cvflow gpu 会说"缺 CUDA 运行库"。
;
; 用 packaging\build.ps1 构建，不要直接双击本文件。

#define AppName "CVFlow"
#define AppPublisher "tanxinji"
#define AppURL "https://github.com/Taning-by/cvflow"
#define AppExe "CVFlow.exe"
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef SrcDir
  #define SrcDir "..\dist\CVFlow"
#endif
#ifndef OutDir
  #define OutDir "..\dist"
#endif

[Setup]
AppId={{8F3C5C2E-9E5B-4D31-9E2B-CVFLOW000001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir={#OutDir}
OutputBaseFilename=CVFlow-Setup-{#AppVersion}-x64
SetupIconFile=..\cvflow\ui\assets\cvflow.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; 只支持 64 位：onnxruntime / PySide6 都没有 32 位轮子
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
PrivilegesRequired=admin
ChangesAssociations=yes

[Languages]
Name: "zh"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full";   Description: "完整安装（含 GPU 推理支持）"
Name: "cpu";    Description: "仅 CPU（不装 CUDA 运行库）"
Name: "custom"; Description: "自定义"; Flags: iscustom

[Components]
Name: "main";     Description: "主程序";        Types: full cpu custom; Flags: fixed
Name: "gpu";      Description: "GPU 推理支持（CUDA 12 + cuDNN 9，约 1.9 GB；需要 NVIDIA 显卡、驱动 527.41 以上）"; Types: full
Name: "examples"; Description: "示例方案与示例图像"; Types: full cpu

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "快捷方式:"
Name: "assoc";       Description: "用 CVFlow 打开 .cvflow 方案文件"; GroupDescription: "文件关联:"
[Files]
; GPU 组件的文件有两处落脚点，两处都要排除/收取：
;   _internal\nvidia\<包>\bin\   spec 里按原布局收的 CUDA / cuDNN 运行库
;   _internal\ 根目录            PyInstaller 解析 provider 依赖时顺手拖进来的那几个
; 所以用文件名模式而不是固定路径——散落在哪都兜得住。
; 这串**不要**抽成 #define：ISPP 的字符串字面量会处理反斜杠转义，写起来容易出错；
; [Files] 行里的路径是普通文本，照抄即可。
; ---- 主程序：除了 GPU 那几样，其余全收 ----
Source: "{#SrcDir}\*"; DestDir: "{app}"; Components: main; Flags: ignoreversion recursesubdirs createallsubdirs; \
    Excludes: "_internal\nvidia\*,_internal\onnxruntime\capi\onnxruntime_providers_cuda.dll,_internal\onnxruntime\capi\onnxruntime_providers_tensorrt.dll,_internal\cudnn*.dll,_internal\cublas*.dll,_internal\cublasLt*.dll,_internal\cudart64*.dll,_internal\cufft*.dll,_internal\curand*.dll,_internal\nvrtc*.dll,_internal\nvJitLink*.dll"
; ---- GPU 组件：CUDA provider + CUDA/cuDNN 运行库 ----
Source: "{#SrcDir}\_internal\onnxruntime\capi\onnxruntime_providers_cuda.dll"; DestDir: "{app}\_internal\onnxruntime\capi"; \
    Components: gpu; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcDir}\_internal\onnxruntime\capi\onnxruntime_providers_tensorrt.dll"; DestDir: "{app}\_internal\onnxruntime\capi"; \
    Components: gpu; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcDir}\_internal\nvidia\*"; DestDir: "{app}\_internal\nvidia"; Components: gpu; \
    Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#SrcDir}\_internal\cudnn*.dll"; DestDir: "{app}\_internal"; Components: gpu; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcDir}\_internal\cublas*.dll"; DestDir: "{app}\_internal"; Components: gpu; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcDir}\_internal\cudart64*.dll"; DestDir: "{app}\_internal"; Components: gpu; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcDir}\_internal\cufft*.dll"; DestDir: "{app}\_internal"; Components: gpu; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcDir}\_internal\curand*.dll"; DestDir: "{app}\_internal"; Components: gpu; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcDir}\_internal\nvrtc*.dll"; DestDir: "{app}\_internal"; Components: gpu; Flags: ignoreversion skipifsourcedoesntexist
Source: "{#SrcDir}\_internal\nvJitLink*.dll"; DestDir: "{app}\_internal"; Components: gpu; Flags: ignoreversion skipifsourcedoesntexist
; ---- 示例：装到可写的 ProgramData，Program Files 下用户改不了 ----
Source: "..\examples\*"; DestDir: "{commonappdata}\{#AppName}\examples"; Components: examples; \
    Flags: ignoreversion recursesubdirs createallsubdirs uninsneveruninstall; Excludes: "__pycache__\*,models\*"

[Dirs]
; 方案、日志、存图都要可写，所以放 ProgramData 而不是 Program Files
Name: "{commonappdata}\{#AppName}";            Permissions: users-modify
Name: "{commonappdata}\{#AppName}\solutions";  Permissions: users-modify
Name: "{commonappdata}\{#AppName}\logs";       Permissions: users-modify
Name: "{commonappdata}\{#AppName}\captures";   Permissions: users-modify

[Icons]
Name: "{group}\{#AppName}";            Filename: "{app}\{#AppExe}"
Name: "{group}\CVFlow 命令行";          Filename: "{app}\cvflow.exe"; Parameters: "--help"
Name: "{group}\检查推理环境";            Filename: "{app}\cvflow.exe"; Parameters: "gpu"
Name: "{group}\卸载 {#AppName}";        Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";      Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.cvflow"; ValueType: string; ValueName: ""; ValueData: "CVFlow.Solution"; \
    Flags: uninsdeletevalue; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\CVFlow.Solution"; ValueType: string; ValueName: ""; ValueData: "CVFlow 方案"; \
    Flags: uninsdeletekey; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\CVFlow.Solution\DefaultIcon"; ValueType: string; ValueName: ""; \
    ValueData: "{app}\{#AppExe},0"; Tasks: assoc
Root: HKA; Subkey: "Software\Classes\CVFlow.Solution\shell\open\command"; ValueType: string; ValueName: ""; \
    ValueData: """{app}\{#AppExe}"" ""%1"""; Tasks: assoc

[Run]
; 装完可选跑一次自检：现场当场知道 CUDA 行不行，而不是等跑流程才发现
Filename: "{app}\cvflow.exe"; Parameters: "gpu"; Description: "检查推理环境（看 CUDA 能不能用）"; \
    Flags: postinstall skipifsilent shellexec
Filename: "{app}\{#AppExe}"; Description: "立即运行 {#AppName}"; Flags: postinstall skipifsilent nowait

[Code]
function NvidiaDriverVersion(): String;
var
  names: TArrayOfString;
  i: Integer;
  ver: String;
begin
  { 显卡驱动版本写在注册表的 DirectX/NVIDIA 项里，nvidia-smi 不一定在 PATH 上 }
  Result := '';
  if RegGetSubkeyNames(HKLM, 'SYSTEM\CurrentControlSet\Services\nvlddmkm', names) then
  begin
    for i := 0 to GetArrayLength(names) - 1 do
    begin
      if RegQueryStringValue(HKLM, 'SYSTEM\CurrentControlSet\Services\nvlddmkm\' + names[i],
                             'NVIDIA_DEV.Version', ver) then
      begin
        Result := ver;
        exit;
      end;
    end;
  end;
end;

function HasNvidiaGpu(): Boolean;
begin
  Result := RegKeyExists(HKLM, 'SYSTEM\CurrentControlSet\Services\nvlddmkm');
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  { 选组件这一页上提醒：机器上没有 NVIDIA 显卡就别勾 GPU，白占 1.9 GB }
  if (CurPageID = wpSelectComponents) and (not HasNvidiaGpu()) then
    MsgBox('没有检测到 NVIDIA 显卡。' + #13#10#13#10 +
           '不勾「GPU 推理支持」可以省下约 1.9 GB，软件照常可用，推理跑在 CPU 上。' + #13#10 +
           '以后换了带显卡的机器，重新运行一次本安装程序勾上即可。',
           mbInformation, MB_OK);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectComponents) and WizardIsComponentSelected('gpu') and (not HasNvidiaGpu()) then
    Result := MsgBox('机器上没有 NVIDIA 显卡，GPU 组件装上也用不了（约 1.9 GB）。仍然要装吗？',
                     mbConfirmation, MB_YESNO) = IDYES;
end;
