"""
Context Trimmer for Forest OS LLM Services.

This module implements a context trimming service that ensures prompts and contexts
sent to LLMs don't exceed token limits while preserving the most relevant information.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
import tiktoken
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)

class TrimmerConfig(BaseModel):
    """Configuration for the ContextTrimmer."""
    max_tokens: int = Field(default=4000, description="Maximum tokens allowed in processed context")
    buffer_tokens: int = Field(default=500, description="Buffer tokens to leave room for response")
    preserve_recent_ratio: float = Field(default=0.6, description="Ratio of tokens to preserve from recent content")
    preserve_first_n_chars: int = Field(default=800, description="Always preserve this many characters from the beginning")
    tiktoken_model: str = Field(default="cl100k_base", description="Tiktoken model to use for encoding")
    section_markers: List[str] = Field(
        default=["##", "===", "---", "*****"],
        description="Markers used to identify logical sections in content"
    )

class ContextSection(BaseModel):
    """A logical section of context with metadata."""
    content: str = Field(..., description="The section content")
    token_count: int = Field(..., description="Number of tokens in this section")
    priority: int = Field(default=0, description="Priority (higher = more important)")
    keep_ratio: float = Field(default=1.0, description="Ratio of section to keep when trimming")
    is_recent: bool = Field(default=False, description="Whether this content is recent (e.g. latest user message)")
    is_system: bool = Field(default=False, description="Whether this is system content (instructions, etc)")
    

class ContextTrimmer:
    """
    Service for trimming context to fit within token limits for LLM requests.
    
    This service analyzes content, divides it into logical sections, and intelligently
    trims it while preserving the most important information to stay within token limits.
    """
    
    def __init__(self, config: Optional[TrimmerConfig] = None):
        """
        Initialize the ContextTrimmer.
        
        Args:
            config: Optional custom configuration for the trimmer
        """
        self.config = config or TrimmerConfig()
        self.encoder = tiktoken.get_encoding(self.config.tiktoken_model)
        logger.info(f"ContextTrimmer initialized with max_tokens={self.config.max_tokens}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            The number of tokens in the text
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))
    
    def identify_sections(self, text: str) -> List[ContextSection]:
        """
        Break text into logical sections based on markers.
        
        Args:
            text: The text to divide into sections
            
        Returns:
            A list of ContextSection objects
        """
        if not text:
            return []
            
        # Create pattern to find section breaks
        pattern = '|'.join(re.escape(marker) for marker in self.config.section_markers)
        
        # If no sections are found, treat the whole text as one section
        if not pattern or not re.search(pattern, text):
            token_count = self.count_tokens(text)
            return [
                ContextSection(
                    content=text,
                    token_count=token_count,
                    priority=1,
                    keep_ratio=1.0
                )
            ]
        
        # Split on section markers
        sections = re.split(f"({pattern}.*(?:\\n|$))", text)
        result = []
        
        current_section = ""
        section_title = ""
        
        for i, section in enumerate(sections):
            # If this is a section marker, it becomes the title for the next section
            if i % 2 == 1:
                section_title = section.strip()
                continue
                
            if section.strip():
                if section_title:
                    content = f"{section_title}\n{section}"
                else:
                    content = section
                    
                token_count = self.count_tokens(content)
                
                # Determine priority - higher for sections with important keywords
                priority = 1
                lower_content = content.lower()
                if any(kw in lower_content for kw in ["summary", "important", "critical", "key"]):
                    priority = 3
                elif any(kw in lower_content for kw in ["context", "background", "detail"]):
                    priority = 2
                
                result.append(
                    ContextSection(
                        content=content,
                        token_count=token_count,
                        priority=priority,
                        keep_ratio=1.0
                    )
                )
                
            section_title = ""
        
        return result
    
    def trim_content(self, content: str, max_tokens: Optional[int] = None) -> Tuple[str, int]:
        """
        Trim content to fit within token limits while preserving important information.
        
        Args:
            content: The content to trim
            max_tokens: Optional custom token limit for this specific trim operation
            
        Returns:
            A tuple containing (trimmed content, token count)
        """
        if not content:
            return "", 0
            
        max_tokens = max_tokens or self.config.max_tokens
        available_tokens = max_tokens - self.config.buffer_tokens
        
        # Quick check if trimming is needed
        token_count = self.count_tokens(content)
        if token_count <= available_tokens:
            return content, token_count

        # Extract section headers to ensure preservation
        section_headers = []
        for marker in self.config.section_markers:
            # Find all section headers using each marker
            import re
            pattern = f"({re.escape(marker)}.*?(?:\n|$))"
            headers = re.findall(pattern, content)
            section_headers.extend([h.strip() for h in headers if h.strip()])
            
            # Also find any standalone headers (like "Section1")
            lines = content.split('\n')
            for line in lines:
                if line.strip() and len(line.strip()) < 30 and not any(c.isspace() for c in line.strip()):
                    if line.strip() not in section_headers:
                        section_headers.append(line.strip())
                        
        # Preserve at least the first section header if possible
        preserved_header = section_headers[0] if section_headers else ""
        preserved_header_tokens = self.count_tokens(preserved_header) if preserved_header else 0
        
        # If we can't fit even one section header, we'll need to truncate it
        if preserved_header and preserved_header_tokens > available_tokens:
            preserved_header = self._truncate_to_token_limit(preserved_header, available_tokens // 2)
            preserved_header_tokens = self.count_tokens(preserved_header)
            
        # Preserve the first n characters as they're usually important
        first_part = content[:self.config.preserve_first_n_chars]
        first_part_tokens = self.count_tokens(first_part)
        
        remaining_tokens = available_tokens - first_part_tokens
        
        # If we can't fit the first part and a section header, prioritize the header
        if remaining_tokens <= 0 and preserved_header:
            # Reduce first_part to make room for the section header
            first_part_max_tokens = available_tokens - preserved_header_tokens
            if first_part_max_tokens <= 0:
                # If we can't fit both, just use the header
                return preserved_header, preserved_header_tokens
            
            first_part = self._truncate_to_token_limit(first_part, first_part_max_tokens)
            first_part_tokens = self.count_tokens(first_part)
            return first_part + "\n" + preserved_header, self.count_tokens(first_part + "\n" + preserved_header)
        elif remaining_tokens <= 0:
            # If no sections and first part is too long, trim it
            logger.warning(f"First part of content exceeds token limit: {first_part_tokens} tokens")
            return self._truncate_to_token_limit(first_part, available_tokens), available_tokens
        
        # Get sections from the rest of the content
        rest = content[self.config.preserve_first_n_chars:]
        sections = self.identify_sections(rest)
        
        # Sort sections by priority (descending)
        sections.sort(key=lambda s: s.priority, reverse=True)
        
        # Calculate how many tokens we can allocate
        remaining_content = []
        used_tokens = first_part_tokens
        
        # If we have a preserved header and it's not in the first part, add it
        if preserved_header and preserved_header not in first_part:
            if used_tokens + preserved_header_tokens <= available_tokens:
                remaining_content.append(preserved_header)
                used_tokens += preserved_header_tokens
        
        for section in sections:
            # Skip if this section is just the header we already preserved
            if preserved_header and section.content.strip() == preserved_header.strip():
                continue
                
            if used_tokens + section.token_count <= available_tokens:
                # Can include the whole section
                remaining_content.append(section.content)
                used_tokens += section.token_count
            else:
                # Need to trim this section
                tokens_for_section = available_tokens - used_tokens
                if tokens_for_section > 20:  # Lower threshold to include more content
                    truncated = self._truncate_to_token_limit(section.content, tokens_for_section)
                    remaining_content.append(truncated)
                    used_tokens = available_tokens
                break
        
        # If we still haven't used the preserved header and have room, add it at the end
        if preserved_header and preserved_header not in first_part and not any(preserved_header in rc for rc in remaining_content):
            if used_tokens + preserved_header_tokens <= available_tokens:
                remaining_content.append(preserved_header)
            elif available_tokens - used_tokens > 0:
                # Add whatever we can fit
                truncated_header = self._truncate_to_token_limit(preserved_header, available_tokens - used_tokens)
                remaining_content.append(truncated_header)
        
        # Combine the result
        trimmed = first_part + "\n".join(remaining_content)
        final_token_count = self.count_tokens(trimmed)
        
        logger.info(f"Trimmed content from {token_count} to {final_token_count} tokens")
        return trimmed, final_token_count
    
    def _truncate_to_token_limit(self, text: str, token_limit: int) -> str:
        """
        Truncate text to fit exactly within a token limit.
        
        Args:
            text: The text to truncate
            token_limit: The maximum number of tokens allowed
            
        Returns:
            The truncated text
        """
        if not text:
            return ""
            
        encoding = self.encoder.encode(text)
        if len(encoding) <= token_limit:
            return text
            
        truncated_encoding = encoding[:token_limit-3]  # Leave room for ellipsis
        truncated = self.encoder.decode(truncated_encoding)
        
        # Add ellipsis to show it was truncated
        return truncated + "..."
    
    def trim_message_array(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Trim an array of chat messages to fit within token limits.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_tokens: Optional custom token limit for this specific trim operation
            
        Returns:
            The trimmed message list
        """
        if not messages:
            return []
            
        max_tokens = max_tokens or self.config.max_tokens
        available_tokens = max_tokens - self.config.buffer_tokens
        
        # Count total tokens in messages
        token_count = 0
        for msg in messages:
            token_count += self.count_tokens(msg.get('content', ''))
            # Account for message metadata (role, etc.) - approximately 4 tokens per message
            token_count += 4
        
        if token_count <= available_tokens:
            return messages
        
        # Preserve system messages and most recent messages
        system_messages = [msg for msg in messages if msg.get('role') == 'system']
        non_system = [msg for msg in messages if msg.get('role') != 'system']
        
        # Count tokens in system messages
        system_tokens = 0
        for msg in system_messages:
            system_tokens += self.count_tokens(msg.get('content', '')) + 4
        
        # Determine tokens available for non-system messages
        available_for_non_system = available_tokens - system_tokens
        
        if available_for_non_system <= 0:
            # Need to trim system messages too
            logger.warning("System messages exceed token limit, trimming required")
            for i, msg in enumerate(system_messages):
                content = msg.get('content', '')
                if content:
                    trimmed, _ = self.trim_content(content, available_tokens // len(system_messages))
                    system_messages[i]['content'] = trimmed
            
            # Recalculate system tokens
            system_tokens = 0
            for msg in system_messages:
                system_tokens += self.count_tokens(msg.get('content', '')) + 4
                
            available_for_non_system = available_tokens - system_tokens
        
        if available_for_non_system <= 0:
            # Even with trimming, system messages take all tokens
            logger.warning("No tokens available for non-system messages")
            return system_messages
        
        # For non-system messages, keep most recent ones
        result = system_messages.copy()
        
        # Start from most recent and work backwards
        tokens_used = system_tokens
        for msg in reversed(non_system):
            content = msg.get('content', '')
            msg_tokens = self.count_tokens(content) + 4
            
            if tokens_used + msg_tokens <= available_tokens:
                # Can include the whole message
                result.append(msg)
                tokens_used += msg_tokens
            else:
                # Need to trim this message
                tokens_for_msg = available_tokens - tokens_used
                if tokens_for_msg > 20:  # Only add if we can include something meaningful
                    trimmed, _ = self.trim_content(content, tokens_for_msg - 4)
                    trimmed_msg = msg.copy()
                    trimmed_msg['content'] = trimmed
                    result.append(trimmed_msg)
                break
        
        # Restore the original message order
        result.sort(key=lambda msg: messages.index(msg) if msg in messages else len(messages))
        
        return result
