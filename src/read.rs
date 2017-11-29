// Copyright 2017 Serde Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{char, cmp, io, str};

use super::error::{Error, ErrorCode, Result};

/// Trait used by the deserializer for iterating over input. This is manually
/// "specialized" for iterating over &[u8]. Once feature(specialization) is
/// stable we can use actual specialization.
///
/// This trait is sealed and cannot be implemented for types outside of
/// `serde_json`.
pub trait Read<'de>: private::Sealed {
    #[doc(hidden)]
    fn next(&mut self) -> io::Result<Option<u8>>;
    #[doc(hidden)]
    fn peek(&mut self) -> io::Result<Option<u8>>;

    /// Only valid after a call to peek(). Discards the peeked byte.
    #[doc(hidden)]
    fn discard(&mut self);

    /// Position of the most recent call to next().
    ///
    /// The most recent call was probably next() and not peek(), but this method
    /// should try to return a sensible result if the most recent call was
    /// actually peek() because we don't always know.
    ///
    /// Only called in case of an error, so performance is not important.
    #[doc(hidden)]
    fn position(&self) -> Position;

    /// Position of the most recent call to peek().
    ///
    /// The most recent call was probably peek() and not next(), but this method
    /// should try to return a sensible result if the most recent call was
    /// actually next() because we don't always know.
    ///
    /// Only called in case of an error, so performance is not important.
    #[doc(hidden)]
    fn peek_position(&self) -> Position;

    /// Offset from the beginning of the input to the next byte that would be
    /// returned by next() or peek().
    #[doc(hidden)]
    fn byte_offset(&self) -> usize;

    /// Assumes the previous byte was a quotation mark. Parses a JSON-escaped
    /// string until the next quotation mark using the given scratch space if
    /// necessary. The scratch space is initially empty.
    #[doc(hidden)]
    fn parse_str<'s, T, F: FnOnce(&str) -> T>(
        &mut self,
        f: F
    ) -> Result<T>;

    /// Assumes the previous byte was a quotation mark. Parses a JSON-escaped
    /// string until the next quotation mark using the given scratch space if
    /// necessary. The scratch space is initially empty.
    ///
    /// This function returns the raw bytes in the string with escape sequences
    /// expanded but without performing unicode validation.
    #[doc(hidden)]
    fn parse_str_raw<'s, T, F: FnOnce(&[u8]) -> T>(
        &mut self,
        f: F,
    ) -> Result<T>;

    /// Assumes the previous byte was a quotation mark. Parses a JSON-escaped
    /// string until the next quotation mark but discards the data.
    #[doc(hidden)]
    fn ignore_str(&mut self) -> Result<()>;

    #[doc(hidden)]
    fn push_scratch(&mut self, u8);
}

#[derive(Clone, Copy)]
pub struct Position {
    pub line: usize,
    pub column: usize,
}

impl Position {
    pub fn update(&mut self, slice: &[u8]) -> usize {
        let mut start_bytes = 0;
        for b in slice {
            if *b == b'\n' {
                start_bytes += self.column + 1;
                self.line += 1;
                self.column = 0;
            } else {
                self.column += 1;
            }
        }
        start_bytes
    }
}

/// JSON input source that reads from a std::io input stream.
pub struct IoRead<R>
where
    R: io::Read,
{
    reader: R,
    /// Temporary storage of peeked byte.
    ch: Option<u8>,
    start_of_line: usize,
    pos: Position,
    scratch: Vec<u8>,
}

/// JSON input source that reads from a slice of bytes.
//
// This is more efficient than other iterators because peek() can be read-only
// and we can compute line/col position only if an error happens.
pub struct SliceRead<'a> {
    slice: &'a [u8],
    /// Index of the *next* byte that will be returned by next() or peek().
    index: usize,
    scratch: Vec<u8>,
}

/// JSON input source that reads from a UTF-8 string.
//
// Able to elide UTF-8 checks by assuming that the input is valid UTF-8.
pub struct StrRead<'a> {
    delegate: SliceRead<'a>,
}

// Prevent users from implementing the Read trait.
mod private {
    pub trait Sealed {}
}

//////////////////////////////////////////////////////////////////////////////

impl<R> IoRead<R>
where
    R: io::Read,
{
    /// Create a JSON input source to read from a std::io input stream.
    pub fn new(reader: R) -> Self {
        IoRead {
            reader,
            ch: None,
            start_of_line: 0,
            pos: Position { line: 1, column: 0 },
            scratch: Vec::with_capacity(128),
        }
    }

    fn read_one_byte(&mut self) -> Option<io::Result<u8>> {
        let mut buf = [0];
        loop {
            return match self.reader.read(&mut buf) {
                Ok(0) => None,
                Ok(..) => {
                    self.start_of_line += self.pos.update(&buf);
                    Some(Ok(buf[0]))
                },
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => Some(Err(e)),
            };
        }
    }
}

impl<R> private::Sealed for IoRead<R>
where
    R: io::Read,
{
}

trait IoReadSpecialization {
    fn parse_str_bytes<'s, T, F>(
            &mut self,
            validate: bool,
            result: F,
        ) -> Result<T>
        where
            F: FnOnce(Position, &[u8]) -> Result<T>;
}

impl<R> IoReadSpecialization for IoRead<R>
where
    R: io::Read,
{
    default fn parse_str_bytes<'s, T, F>(
        &mut self,
        validate: bool,
        result: F,
    ) -> Result<T>
    where
        F: FnOnce(Position, &[u8]) -> Result<T>,
    {
        self.scratch.clear();
        loop {
            let ch = try!(next_or_eof(self));
            if !ESCAPE[ch as usize] {
                self.scratch.push(ch);
                continue;
            }
            match ch {
                b'"' => {
                    return result(self.pos, &self.scratch);
                }
                b'\\' => {
                    try!(parse_escape(self));
                }
                _ => {
                    if validate {
                        return error(self.position(), ErrorCode::InvalidUnicodeCodePoint);
                    }
                    self.scratch.push(ch);
                }
            }
        }
    }
}

impl<R> IoReadSpecialization for IoRead<R>
where
    R: io::BufRead,
{
    // If we read the whole string into the buffer at once, we can avoid
    // pushing into the scratch space
    fn parse_str_bytes<'s, T, F>(
        &mut self,
        validate: bool,
        result: F,
    ) -> Result<T>
    where
        F: FnOnce(Position, &[u8]) -> Result<T>,
    {
        let ret;
        let last_consume;

        // This loop is just used for jumping in the fast path
        loop {
            let mut ch;
            let consumed = {
                let available = match self.reader.fill_buf() {
                    Ok(n) => n,
                    Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(Error::io(e)),
                };

                // Fast path to zero copies
                // TODO(bouk): can be deduped/simplified
                match available.iter().position(|b| ESCAPE[*b as usize]) {
                    // In this case we extend/callback with everything except for the last byte
                    Some(i) => {
                        let b = available[i];
                        if b == b'"' {
                            // Here we are doing a callback without copying anything into scratch, and just using
                            // the buffer of the io::BufRead to do the callback. v fast
                            self.start_of_line += self.pos.update(&available[..i + 1]);
                            ret = result(self.pos, &available[..i]);
                            last_consume = i + 1;
                            // We need to break here because we can only call result once, so we need to break
                            // out of the loop to tell the Rust borrow checker that we won't
                            break;
                        } else {
                            // Now we need to copy everything read so far into scratch, and do normal reading
                            self.start_of_line += self.pos.update(&available[..i]);
                            self.scratch.clear();
                            self.scratch.extend_from_slice(&available[..i]);
                            ch = Some(b);
                            i + 1
                        }
                    },
                    // In this case we put the whole buffer into scratch and do regular reading
                    None => {
                        // Reached end of buffer, fall back to normal reading
                        self.start_of_line += self.pos.update(available);
                        self.scratch.clear();
                        self.scratch.extend_from_slice(available);
                        ch = None;
                        available.len()
                    },
                }
            };
            self.reader.consume(consumed);

            loop {
                match ch.take() {
                    Some(b'"') => {
                        return result(self.pos, &self.scratch);
                    }
                    Some(b'\\') => {
                        try!(parse_escape(self));
                    }
                    Some(ch) => {
                        if validate {
                            return error(self.pos, ErrorCode::InvalidUnicodeCodePoint);
                        }
                        self.scratch.push(ch);
                    }
                    None => {
                        let consumed = {
                            let available = match self.reader.fill_buf() {
                                Ok(n) => n,
                                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                                Err(e) => return Err(Error::io(e)),
                            };

                            match available.iter().position(|b| ESCAPE[*b as usize]) {
                                Some(i) => {
                                    self.start_of_line += self.pos.update(&available[..i]);
                                    self.scratch.extend_from_slice(&available[..i]);
                                    ch = Some(available[i]);
                                    i + 1
                                },
                                None => {
                                    self.start_of_line += self.pos.update(available);
                                    self.scratch.extend_from_slice(available);
                                    ch = None;
                                    available.len()
                                },
                            }
                        };
                        self.reader.consume(consumed);
                    }
                }
            }
        }
        self.reader.consume(last_consume);
        ret
    }
}

impl<'de, R> Read<'de> for IoRead<R>
where
    R: io::Read,
{
    #[inline]
    fn next(&mut self) -> io::Result<Option<u8>> {
        match self.ch.take() {
            Some(ch) => Ok(Some(ch)),
            None => {
                match self.read_one_byte() {
                    Some(Err(err)) => Err(err),
                    Some(Ok(ch)) => Ok(Some(ch)),
                    None => Ok(None),
                }
            }
        }
    }

    #[inline]
    fn peek(&mut self) -> io::Result<Option<u8>> {
        match self.ch {
            Some(ch) => Ok(Some(ch)),
            None => {
                match self.read_one_byte() {
                    Some(Err(err)) => Err(err),
                    Some(Ok(ch)) => {
                        self.ch = Some(ch);
                        Ok(self.ch)
                    }
                    None => Ok(None),
                }
            }
        }
    }

    #[inline]
    fn discard(&mut self) {
        self.ch = None;
    }

    fn position(&self) -> Position {
        self.pos
    }

    fn peek_position(&self) -> Position {
        // The LineColIterator updates its position during peek() so it has the
        // right one here.
        self.position()
    }

    fn byte_offset(&self) -> usize {
        match self.ch {
            Some(_) => self.start_of_line + self.pos.column + 1,
            None => self.start_of_line + self.pos.column,
        }
    }

    fn parse_str<'s, T, F: FnOnce(&str) -> T>(
        &mut self,
        f: F
    ) -> Result<T> {
        self.parse_str_bytes(true, |pos, s| as_str(pos, s).map(|s| f(s)))
    }

    fn parse_str_raw<'s, T, F: FnOnce(&[u8]) -> T>(
        &mut self,
        f: F
    ) -> Result<T> {
        self.parse_str_bytes(false, |_, bytes| Ok(f(bytes)))
    }

    fn ignore_str(&mut self) -> Result<()> {
        loop {
            let ch = try!(next_or_eof(self));
            if !ESCAPE[ch as usize] {
                continue;
            }
            match ch {
                b'"' => {
                    return Ok(());
                }
                b'\\' => {
                    try!(ignore_escape(self));
                }
                _ => {
                    return error(self.position(), ErrorCode::InvalidUnicodeCodePoint);
                }
            }
        }
    }

    #[inline]
    fn push_scratch(&mut self, b: u8) {
        self.scratch.push(b);
    }
}

//////////////////////////////////////////////////////////////////////////////

impl<'a> SliceRead<'a> {
    /// Create a JSON input source to read from a slice of bytes.
    pub fn new(slice: &'a [u8]) -> Self {
        SliceRead {
            slice: slice,
            index: 0,
            scratch: Vec::with_capacity(128),
        }
    }

    fn position_of_index(&self, i: usize) -> Position {
        let mut pos = Position { line: 1, column: 0 };
        pos.update(&self.slice[..i]);
        pos
    }

    /// The big optimization here over IoRead is that if the string contains no
    /// backslash escape sequences, the returned &str is a slice of the raw JSON
    /// data so we avoid copying into the scratch space.
    fn parse_str_bytes<'s, T, F>(
        &mut self,
        validate: bool,
        result: F,
    ) -> Result<T>
    where
        F: FnOnce(Position, &[u8]) -> Result<T>,
    {
        self.scratch.clear();
        // Index of the first byte not yet copied into the scratch space.
        let mut start = self.index;
        loop {
            while self.index < self.slice.len() && !ESCAPE[self.slice[self.index] as usize] {
                self.index += 1;
            }
            if self.index == self.slice.len() {
                return error(self.position(), ErrorCode::EofWhileParsingString);
            }
            match self.slice[self.index] {
                b'"' => {
                    if self.scratch.is_empty() {
                        // Fast path: return a slice of the raw JSON without any
                        // copying.
                        let borrowed = &self.slice[start..self.index];
                        self.index += 1;
                        return result(self.position(), borrowed);
                    } else {
                        self.scratch.extend_from_slice(&self.slice[start..self.index]);
                        // "as &[u8]" is required for rustc 1.8.0
                        let copied = &self.scratch as &[u8];
                        self.index += 1;
                        return result(self.position(), copied);
                    }
                }
                b'\\' => {
                    self.scratch.extend_from_slice(&self.slice[start..self.index]);
                    self.index += 1;
                    try!(parse_escape(self));
                    start = self.index;
                }
                _ => {
                    if validate {
                        return error(self.position(), ErrorCode::InvalidUnicodeCodePoint);
                    }
                    self.index += 1;
                }
            }
        }
    }
}

impl<'a> private::Sealed for SliceRead<'a> {}

impl<'a> Read<'a> for SliceRead<'a> {
    #[inline]
    fn next(&mut self) -> io::Result<Option<u8>> {
        // `Ok(self.slice.get(self.index).map(|ch| { self.index += 1; *ch }))`
        // is about 10% slower.
        Ok(
            if self.index < self.slice.len() {
                let ch = self.slice[self.index];
                self.index += 1;
                Some(ch)
            } else {
                None
            },
        )
    }

    #[inline]
    fn peek(&mut self) -> io::Result<Option<u8>> {
        // `Ok(self.slice.get(self.index).map(|ch| *ch))` is about 10% slower
        // for some reason.
        Ok(
            if self.index < self.slice.len() {
                Some(self.slice[self.index])
            } else {
                None
            },
        )
    }

    #[inline]
    fn discard(&mut self) {
        self.index += 1;
    }

    fn position(&self) -> Position {
        self.position_of_index(self.index)
    }

    fn peek_position(&self) -> Position {
        // Cap it at slice.len() just in case the most recent call was next()
        // and it returned the last byte.
        self.position_of_index(cmp::min(self.slice.len(), self.index + 1))
    }

    fn byte_offset(&self) -> usize {
        self.index
    }

    fn parse_str<'s, T, F: FnOnce(&str) -> T>(
        &mut self,
        f: F
    ) -> Result<T> {
        self.parse_str_bytes(true, |pos, bytes| as_str(pos, bytes).map(|s| f(s)))
    }

    fn parse_str_raw<'s, T, F: FnOnce(&[u8]) -> T>(
        &mut self,
        f: F,
    ) -> Result<T> {
        self.parse_str_bytes(false, |_, bytes| Ok(f(bytes)))
    }

    fn ignore_str(&mut self) -> Result<()> {
        loop {
            while self.index < self.slice.len() && !ESCAPE[self.slice[self.index] as usize] {
                self.index += 1;
            }
            if self.index == self.slice.len() {
                return error(self.position(), ErrorCode::EofWhileParsingString);
            }
            match self.slice[self.index] {
                b'"' => {
                    self.index += 1;
                    return Ok(());
                }
                b'\\' => {
                    self.index += 1;
                    try!(ignore_escape(self));
                }
                _ => {
                    return error(self.position(), ErrorCode::InvalidUnicodeCodePoint);
                }
            }
        }
    }

    #[inline]
    fn push_scratch(&mut self, b: u8) {
        self.scratch.push(b);
    }
}

//////////////////////////////////////////////////////////////////////////////

impl<'a> StrRead<'a> {
    /// Create a JSON input source to read from a UTF-8 string.
    pub fn new(s: &'a str) -> Self {
        StrRead { delegate: SliceRead::new(s.as_bytes()) }
    }
}

impl<'a> private::Sealed for StrRead<'a> {}

impl<'a> Read<'a> for StrRead<'a> {
    #[inline]
    fn next(&mut self) -> io::Result<Option<u8>> {
        self.delegate.next()
    }

    #[inline]
    fn peek(&mut self) -> io::Result<Option<u8>> {
        self.delegate.peek()
    }

    #[inline]
    fn discard(&mut self) {
        self.delegate.discard();
    }

    fn position(&self) -> Position {
        self.delegate.position()
    }

    fn peek_position(&self) -> Position {
        self.delegate.peek_position()
    }

    fn byte_offset(&self) -> usize {
        self.delegate.byte_offset()
    }

    fn parse_str<'s, T, F: FnOnce(&str) -> T>(
        &mut self,
        f: F,
    ) -> Result<T> {
        self.delegate
            .parse_str_bytes(true, |_, bytes| {
                // The input is assumed to be valid UTF-8 and the \u-escapes are
                // checked along the way, so don't need to check here.
                Ok(f(unsafe { str::from_utf8_unchecked(bytes) }))
            })
    }

    fn parse_str_raw<'s, T, F: FnOnce(&[u8]) -> T>(
        &mut self,
        f: F,
    ) -> Result<T> {
        self.delegate.parse_str_raw(f)
    }

    fn ignore_str(&mut self) -> Result<()> {
        self.delegate.ignore_str()
    }

    #[inline]
    fn push_scratch(&mut self, b: u8) {
        self.delegate.scratch.push(b);
    }
}

//////////////////////////////////////////////////////////////////////////////

const CT: bool = true; // control character \x00...\x1F
const QU: bool = true; // quote \x22
const BS: bool = true; // backslash \x5C
const O: bool = false; // allow unescaped

// Lookup table of bytes that must be escaped. A value of true at index i means
// that byte i requires an escape sequence in the input.
#[cfg_attr(rustfmt, rustfmt_skip)]
static ESCAPE: [bool; 256] = [
    //   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
    CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, // 0
    CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, CT, // 1
     O,  O, QU,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // 2
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // 3
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // 4
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, BS,  O,  O,  O, // 5
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // 6
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // 7
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // 8
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // 9
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // A
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // B
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // C
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // D
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // E
     O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O,  O, // F
];

fn next_or_eof<'de, R: ?Sized + Read<'de>>(read: &mut R) -> Result<u8> {
    match try!(read.next().map_err(Error::io)) {
        Some(b) => Ok(b),
        None => error(read.position(), ErrorCode::EofWhileParsingString),
    }
}

fn error<'de, T>(pos: Position, reason: ErrorCode) -> Result<T> {
    Err(Error::syntax(reason, pos.line, pos.column))
}

fn as_str<'de, 's>(pos: Position, slice: &'s [u8]) -> Result<&'s str> {
    str::from_utf8(slice).or_else(|_| error(pos, ErrorCode::InvalidUnicodeCodePoint))
}

/// Parses a JSON escape sequence and appends it into the scratch space. Assumes
/// the previous byte read was a backslash.
fn parse_escape<'de, R: Read<'de>>(read: &mut R) -> Result<()> {
    let ch = try!(next_or_eof(read));

    match ch {
        b'"' => read.push_scratch(b'"'),
        b'\\' => read.push_scratch(b'\\'),
        b'/' => read.push_scratch(b'/'),
        b'b' => read.push_scratch(b'\x08'),
        b'f' => read.push_scratch(b'\x0c'),
        b'n' => read.push_scratch(b'\n'),
        b'r' => read.push_scratch(b'\r'),
        b't' => read.push_scratch(b'\t'),
        b'u' => {
            let c = match try!(decode_hex_escape(read)) {
                0xDC00...0xDFFF => {
                    return error(read.position(), ErrorCode::LoneLeadingSurrogateInHexEscape);
                }

                // Non-BMP characters are encoded as a sequence of
                // two hex escapes, representing UTF-16 surrogates.
                n1 @ 0xD800...0xDBFF => {
                    if try!(next_or_eof(read)) != b'\\' {
                        return error(read.position(), ErrorCode::UnexpectedEndOfHexEscape);
                    }
                    if try!(next_or_eof(read)) != b'u' {
                        return error(read.position(), ErrorCode::UnexpectedEndOfHexEscape);
                    }

                    let n2 = try!(decode_hex_escape(read));

                    if n2 < 0xDC00 || n2 > 0xDFFF {
                        return error(read.position(), ErrorCode::LoneLeadingSurrogateInHexEscape);
                    }

                    let n = (((n1 - 0xD800) as u32) << 10 | (n2 - 0xDC00) as u32) + 0x1_0000;

                    match char::from_u32(n) {
                        Some(c) => c,
                        None => {
                            return error(read.position(), ErrorCode::InvalidUnicodeCodePoint);
                        }
                    }
                }

                n => {
                    match char::from_u32(n as u32) {
                        Some(c) => c,
                        None => {
                            return error(read.position(), ErrorCode::InvalidUnicodeCodePoint);
                        }
                    }
                }
            };


            let mut buf = [0; 4];
            c.encode_utf8(&mut buf);
            for b in buf[..c.len_utf8()].iter() {
                read.push_scratch(*b);
            }
        }
        _ => {
            return error(read.position(), ErrorCode::InvalidEscape);
        }
    }

    Ok(())
}

/// Parses a JSON escape sequence and discards the value. Assumes the previous
/// byte read was a backslash.
fn ignore_escape<'de, R: ?Sized + Read<'de>>(read: &mut R) -> Result<()> {
    let ch = try!(next_or_eof(read));

    match ch {
        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
        b'u' => {
            let n = match try!(decode_hex_escape(read)) {
                0xDC00...0xDFFF => {
                    return error(read.position(), ErrorCode::LoneLeadingSurrogateInHexEscape);
                }

                // Non-BMP characters are encoded as a sequence of
                // two hex escapes, representing UTF-16 surrogates.
                n1 @ 0xD800...0xDBFF => {
                    if try!(next_or_eof(read)) != b'\\' {
                        return error(read.position(), ErrorCode::UnexpectedEndOfHexEscape);
                    }
                    if try!(next_or_eof(read)) != b'u' {
                        return error(read.position(), ErrorCode::UnexpectedEndOfHexEscape);
                    }

                    let n2 = try!(decode_hex_escape(read));

                    if n2 < 0xDC00 || n2 > 0xDFFF {
                        return error(read.position(), ErrorCode::LoneLeadingSurrogateInHexEscape);
                    }

                    (((n1 - 0xD800) as u32) << 10 | (n2 - 0xDC00) as u32) + 0x1_0000
                }

                n => n as u32,
            };

            if char::from_u32(n).is_none() {
                return error(read.position(), ErrorCode::InvalidUnicodeCodePoint);
            }
        }
        _ => {
            return error(read.position(), ErrorCode::InvalidEscape);
        }
    }

    Ok(())
}

fn decode_hex_escape<'de, R: ?Sized + Read<'de>>(read: &mut R) -> Result<u16> {
    let mut n = 0;
    for _ in 0..4 {
        n = match try!(next_or_eof(read)) {
            c @ b'0'...b'9' => n * 16_u16 + ((c as u16) - (b'0' as u16)),
            b'a' | b'A' => n * 16_u16 + 10_u16,
            b'b' | b'B' => n * 16_u16 + 11_u16,
            b'c' | b'C' => n * 16_u16 + 12_u16,
            b'd' | b'D' => n * 16_u16 + 13_u16,
            b'e' | b'E' => n * 16_u16 + 14_u16,
            b'f' | b'F' => n * 16_u16 + 15_u16,
            _ => {
                return error(read.position(), ErrorCode::InvalidEscape);
            }
        };
    }
    Ok(n)
}
