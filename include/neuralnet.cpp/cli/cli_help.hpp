// ── cli_help.hpp — 各 CLI `--help` 输出的统一排版 ──────────────────────────
//
// 目标：所有 CLI 的帮助共用同一套版式与列宽算法，改文案只改表、不手对空格。
//
// 统一版式（各程序一致）：
//   <标题>
//
//   用法:
//     <prog> <参数...>     <说明>
//
//   <位置参数节>:                      ← 仅当有位置参数时出现
//     <name>               <说明>
//
//   选项:
//     --flag <arg>         <说明>
//                          <描述的续行（desc 内的 '\n'）>
//     --help, -h           显示此帮助信息
//
//   <自由节>:
//     --flag <arg>         <说明>
//     说明性段落（note/text，4/2 空格缩进）
//
// 列宽规则：描述列 = 2 + left + 2，left = 全部条目名的最大**显示宽度**
// 再夹在 [16, 34]，且**跨节统一**（不会被某一节的短条目带偏）；条目名超出
// left 时描述另起一行并对齐到描述列。显示宽度按终端列算：ASCII 记 1，
// CJK/全角记 2（`--gpu [索引|名称]` 这类含中文的条目名才能与纯 ASCII 条目
// 对齐）。用法行单独计算列宽（上限 56），超出时说明行缩进 4 列另起一行。
//
// ⚠ 审计约定（bench/gui_cli_audit.py §[2] 幽灵选项检测依赖）：
//   帮助内容一律写成 `help.<方法>(...)`，且**每条调用必须落在同一物理行**
//   （长描述用同一字符串里的 '\n' 续行，不要跨行写实参）。审计把
//   `help.<方法>(` 行排除在"解析分支"之外，从而保留"帮助声明了、解析不存在"
//   的检测能力。新增帮助时请沿用该写法，否则门禁会失效。
// ────────────────────────────────────────────────────────────────────────────

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace nn::cli
{
    // ── 终端显示宽度：ASCII 记 1 列，CJK/全角记 2 列 ─────────────────────
    // 覆盖常见东亚宽字符区段（含全角标点、CJK 扩展）；非法 UTF-8 字节按 1 列。
    [[nodiscard]] inline bool is_wide_codepoint(std::uint32_t cp) noexcept
    {
        if (cp >= 0x1100u && cp <= 0x115Fu) return true;   // 谚文字母
        if (cp >= 0x2E80u && cp <= 0x303Eu) return true;   // CJK 部首/符号/假名
        if (cp >= 0x3041u && cp <= 0x33FFu) return true;   // 假名/注音/CJK 兼容
        if (cp >= 0x3400u && cp <= 0x4DBFu) return true;   // CJK 扩展 A
        if (cp >= 0x4E00u && cp <= 0x9FFFu) return true;   // CJK 统一表意
        if (cp >= 0xA000u && cp <= 0xA4CFu) return true;   // 彝文
        if (cp >= 0xAC00u && cp <= 0xD7A3u) return true;   // 谚文音节
        if (cp >= 0xF900u && cp <= 0xFAFFu) return true;   // CJK 兼容表意
        if (cp >= 0xFE30u && cp <= 0xFE4Fu) return true;   // CJK 兼容形式
        if (cp >= 0xFF00u && cp <= 0xFF60u) return true;   // 全角形式
        if (cp >= 0xFFE0u && cp <= 0xFFE6u) return true;   // 全角符号
        if (cp >= 0x20000u && cp <= 0x3FFFD) return true;  // CJK 扩展 B+
        return false;
    }

    [[nodiscard]] inline std::size_t display_width(std::string_view s) noexcept
    {
        std::size_t width = 0;
        for (std::size_t i = 0; i < s.size();)
        {
            const auto lead = static_cast<unsigned char>(s[i]);
            std::size_t len = 1;
            if (lead >= 0xF0u) len = 4;
            else if (lead >= 0xE0u) len = 3;
            else if (lead >= 0xC0u) len = 2;

            std::uint32_t cp = lead;
            bool valid = (i + len <= s.size());
            if (valid && len > 1)
            {
                cp = lead & (0xFFu >> (len + 1));
                for (std::size_t k = 1; k < len; ++k)
                {
                    const auto cont = static_cast<unsigned char>(s[i + k]);
                    if ((cont & 0xC0u) != 0x80u) { valid = false; break; }
                    cp = (cp << 6) | (cont & 0x3Fu);
                }
            }
            width += (valid && is_wide_codepoint(cp)) ? 2 : 1;
            i += valid ? len : 1;
        }
        return width;
    }

    // ── 帮助排版器 ───────────────────────────────────────────────────────
    // 用法：在 print_usage 里构造局部对象，链式填表；析构时自动渲染一次
    // （也可显式 flush()）。每条调用写在同一物理行（见文件头审计约定）。
    class Help
    {
    public:
        Help(std::ostream &out, std::string_view prog, std::string_view title)
            : out_(out), prog_(prog), title_(title)
        {
        }

        ~Help() { flush(); }

        Help(const Help &) = delete;
        Help &operator=(const Help &) = delete;

        // 用法行：渲染为 `  <prog> <args>`；desc 非空时对齐到用法列。
        Help &usage(std::string_view args, std::string_view desc = {})
        {
            usage_.push_back(Row{Kind::Entry, prefix_prog(std::string(args)), std::string(desc)});
            return *this;
        }

        // 开一个新节（标题后自动空行）
        Help &section(std::string_view title)
        {
            sections_.push_back(Section{std::string(title), {}});
            return *this;
        }

        // 节内说明段落，4 空格缩进
        Help &note(std::string_view line)
        {
            current().push_back(Row{Kind::Note, std::string(line), {}});
            return *this;
        }

        // 自由文本行，2 空格缩进
        Help &text(std::string_view line)
        {
            current().push_back(Row{Kind::Text, std::string(line), {}});
            return *this;
        }

        // 位置参数 / 词条（无 `--` 前缀），与选项同列对齐
        Help &item(std::string_view name, std::string_view desc)
        {
            current().push_back(Row{Kind::Entry, std::string(name), std::string(desc)});
            return *this;
        }

        // 选项：flags 例如 "--gpu [索引|名称]"；desc 内的 '\n' 为续行
        Help &opt(std::string_view flags, std::string_view desc)
        {
            current().push_back(Row{Kind::Entry, std::string(flags), std::string(desc)});
            return *this;
        }

        void flush()
        {
            if (rendered_) return;
            rendered_ = true;

            out_ << title_ << "\n";

            if (!usage_.empty())
            {
                out_ << "\n用法:\n";
                const std::size_t usage_col = column_for(usage_, kMaxUsageWidth);
                for (const Row &r : usage_)
                    write_entry(r.name, r.desc, usage_col, kMaxUsageWidth, kUsageOverflow);
            }

            // 选项列宽全局统一（跨节一致），不被某一节的短条目带偏
            std::vector<Row> entries;
            for (const Section &s : sections_)
                for (const Row &r : s.rows)
                    if (r.kind == Kind::Entry)
                        entries.push_back(r);
            const std::size_t col = column_for(entries, kMaxOptionWidth);

            for (const Section &s : sections_)
            {
                if (!s.title.empty()) out_ << "\n" << s.title << ":\n";
                for (const Row &r : s.rows)
                {
                    switch (r.kind)
                    {
                    case Kind::Entry:
                        write_entry(r.name, r.desc, col, kMaxOptionWidth, col);
                        break;
                    case Kind::Note:
                        write_paragraph(r.name, kNoteIndent);
                        break;
                    case Kind::Text:
                        write_paragraph(r.name, kTextIndent);
                        break;
                    }
                }
            }
        }

    private:
        enum Kind { Entry, Note, Text };

        struct Row
        {
            Kind kind;
            std::string name;
            std::string desc;
        };

        struct Section
        {
            std::string title;
            std::vector<Row> rows;
        };

        static constexpr std::size_t kMinColumn = 16;   // 条目名最小列宽
        static constexpr std::size_t kMaxOptionWidth = 34;  // 选项名最大列宽
        static constexpr std::size_t kMaxUsageWidth = 56;   // 用法行最大列宽
        static constexpr std::size_t kUsageOverflow = 4;    // 用法行过长时说明的缩进
        static constexpr std::size_t kNoteIndent = 4;
        static constexpr std::size_t kTextIndent = 2;

        std::vector<Row> &current()
        {
            if (sections_.empty()) section("");
            return sections_.back().rows;
        }

        [[nodiscard]] std::string prefix_prog(const std::string &args) const
        {
            std::string left = prog_;
            if (!args.empty())
            {
                left += ' ';
                left += args;
            }
            return left;
        }

        // 描述列 = 2（缩进）+ left + 2（间隔），left 夹在 [kMinColumn, max]
        [[nodiscard]] static std::size_t column_for(
            const std::vector<Row> &rows, std::size_t max_width) noexcept
        {
            std::size_t left = kMinColumn;
            for (const Row &r : rows)
                if (r.kind == Entry)
                    left = std::max(left, display_width(r.name));
            left = std::min(left, max_width);
            return 2 + left + 2;
        }

        // 段落：每行统一缩进 indent（note/text 用）
        void write_paragraph(const std::string &body, std::size_t indent)
        {
            const std::string pad(indent, ' ');
            std::size_t pos = 0;
            while (true)
            {
                const std::size_t nl = body.find('\n', pos);
                const std::string_view part =
                    (nl == std::string::npos) ? std::string_view(body).substr(pos)
                                              : std::string_view(body).substr(pos, nl - pos);
                out_ << pad << part << "\n";
                if (nl == std::string::npos) break;
                pos = nl + 1;
            }
        }

        void write_entry(const std::string &name, const std::string &desc,
                         std::size_t desc_col, std::size_t max_width,
                         std::size_t over_indent)
        {
            out_ << "  " << name;
            if (desc.empty())
            {
                out_ << "\n";
                return;
            }

            const std::size_t w = display_width(name);
            const bool same_line = (w <= max_width);
            if (same_line)
                out_ << std::string(desc_col - 2 - w, ' ');
            else
                out_ << "\n" << std::string(over_indent, ' ');

            std::size_t pos = 0;
            bool first = true;
            while (true)
            {
                const std::size_t nl = desc.find('\n', pos);
                const std::string_view part =
                    (nl == std::string::npos) ? std::string_view(desc).substr(pos)
                                              : std::string_view(desc).substr(pos, nl - pos);
                if (!first) out_ << std::string(over_indent, ' ');
                out_ << part << "\n";
                first = false;
                if (nl == std::string::npos) break;
                pos = nl + 1;
            }
        }

        std::ostream &out_;
        std::string prog_;
        std::string title_;
        std::vector<Row> usage_;
        std::vector<Section> sections_;
        bool rendered_ = false;
    };
} // namespace nn::cli
