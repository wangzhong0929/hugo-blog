#!/bin/bash

# 域名Nginx配置脚本
echo "🌐 配置域名和Nginx..."

# 获取用户输入的域名
read -p "请输入你的域名（例如：example.com）: " DOMAIN

if [ -z "$DOMAIN" ]; then
    echo "❌ 域名不能为空"
    exit 1
fi

echo "📝 配置域名: $DOMAIN"

# 1. 备份现有配置
echo "💾 备份现有配置..."
sudo cp /etc/nginx/sites-available/blog /etc/nginx/sites-available/blog.backup.$(date +%Y%m%d_%H%M%S)

# 2. 创建新的Nginx配置
echo "🔧 创建Nginx配置..."
sudo tee /etc/nginx/sites-available/blog > /dev/null << EOF
server {
    listen 80;
    listen [::]:80;
    
    server_name $DOMAIN www.$DOMAIN;
    
    root /var/www/blog/myblog/public;
    index index.html index.htm;
    
    # 日志文件
    access_log /var/log/nginx/${DOMAIN}_access.log;
    error_log /var/log/nginx/${DOMAIN}_error.log;
    
    # 主要内容
    location / {
        try_files \$uri \$uri/ =404;
    }
    
    # 静态文件缓存
    location ~* \.(css|js|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1M;
        add_header Cache-Control "public, immutable";
        add_header Vary Accept-Encoding;
    }
    
    # Gzip压缩
    location ~* \.(css|js|html|xml|txt)$ {
        gzip on;
        gzip_vary on;
        gzip_min_length 1024;
        gzip_comp_level 6;
        gzip_types
            text/plain
            text/css
            text/xml
            text/javascript
            application/javascript
            application/xml+rss
            application/json;
    }
    
    # 安全设置
    location ~ /\. {
        deny all;
    }
    
    # 防止访问敏感文件
    location ~* \.(bak|config|sql|fla|psd|ini|log|sh|inc|swp|dist)$ {
        deny all;
    }
    
    # 安全头部
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # 错误页面
    error_page 404 /404.html;
    error_page 500 502 503 504 /50x.html;
}
EOF

# 3. 更新Hugo配置
echo "📚 更新Hugo配置..."
cd /var/www/blog/myblog

# 备份原配置
cp hugo.toml hugo.toml.backup.$(date +%Y%m%d_%H%M%S)

# 更新baseURL
sed -i "s|baseURL = 'http://your-domain.com'|baseURL = 'http://$DOMAIN'|g" hugo.toml

# 如果配置文件中没有baseURL，添加它
if ! grep -q "baseURL" hugo.toml; then
    sed -i "1i baseURL = 'http://$DOMAIN'" hugo.toml
fi

# 4. 重新生成Hugo站点
echo "🏗️ 重新生成Hugo站点..."
hugo

# 5. 设置权限
sudo chown -R www-data:www-data public/
sudo chmod -R 755 public/

# 6. 测试Nginx配置
echo "🧪 测试Nginx配置..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "✅ Nginx配置测试通过"
    
    # 7. 重新加载Nginx
    echo "🔄 重新加载Nginx..."
    sudo systemctl reload nginx
    
    echo "✅ 配置完成！"
else
    echo "❌ Nginx配置有错误"
    exit 1
fi

# 8. 检查防火墙
echo "🔥 检查防火墙设置..."
sudo ufw status

# 确保HTTP端口开放
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 9. 提供下一步指引
echo ""
echo "🎉 配置完成！"
echo ""
echo "📋 接下来的步骤："
echo "1. 在NameSilo中设置DNS解析："
echo "   - A记录: @ -> $(curl -s ifconfig.me)"
echo "   - CNAME记录: www -> @"
echo ""
echo "2. 等待DNS传播（几分钟到几小时）"
echo ""
echo "3. 测试访问："
echo "   - http://$DOMAIN"
echo "   - http://www.$DOMAIN"
echo ""
echo "4. 可选：安装SSL证书"
echo "   sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN"
echo ""
echo "🔍 检查DNS解析状态："
echo "nslookup $DOMAIN"
echo "nslookup www.$DOMAIN"
